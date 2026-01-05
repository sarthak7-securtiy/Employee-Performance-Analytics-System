from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import json
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename
import statistics
import sys
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from models.database import init_db, User, Test, Answer, Evaluation, Attendance, Feedback
from utils.ocr_extractor import OCRExtractor
from utils.evaluation_engine import EvaluationEngine

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize database engine (but not session)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use the database URI from config
engine = create_engine(app.config['DATABASE_URI'])
Session = sessionmaker(bind=engine)

# Initialize utilities
ocr_extractor = OCRExtractor()
evaluation_engine = EvaluationEngine()

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Helper function to serialize database objects
def serialize_object(obj):
    """Convert database object to dictionary"""
    if hasattr(obj, '__dict__'):
        result = obj.__dict__.copy()
        # Remove SQLAlchemy internal attributes
        result.pop('_sa_instance_state', None)
        # Convert datetime objects to strings
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result
    return obj

# Helper function to get database session
def get_db_session():
    """Create and return a new database session"""
    return Session()

# Routes

@app.route('/')
def home():
    # Serve the frontend HTML file
    return send_file(os.path.join(app.root_path, '..', 'static', 'index.html'))
@app.route('/api')
def api_info():
    return jsonify({
        "message": "Tata Motors Employee Performance Analytics API",
        "version": "1.0",
        "endpoints": {
            "POST /api/upload": "Upload test papers (PDF/JPG/PNG)",
            "GET /api/tests": "Get all tests",
            "GET /api/tests/<test_id>": "Get specific test details",
            "GET /api/trainings": "Get all unique training names",
            "GET /api/faculties": "Get all unique faculty names",
            "GET /api/analytics/batch/<test_id>": "Get batch analytics",
            "GET /api/analytics/candidate/<candidate_id>": "Get candidate analytics",
            "GET /api/analytics/overall": "Get overall analytics across all tests",
            "GET /api/analytics/combined-pass-rate": "Get combined pass rate across all trainings",
            "GET /api/analytics/historical/training/<training_name>": "Get historical analytics for a specific training name",
            "GET /api/analytics/historical/date-range?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD": "Get historical analytics for a date range",
            "GET /api/analytics/historical/faculty/<faculty_name>": "Get historical analytics for a specific faculty name",
            "GET /api/analytics/historical/filtered?training_name=&start_date=&end_date=&faculty_name=": "Get historical analytics with multiple filters",
            "GET /api/reports/batch/<test_id>": "Generate batch report",
            "GET /api/reports/candidate/<candidate_id>": "Generate candidate report"
        }
    })
@app.route('/api/upload', methods=['POST'])
def upload_test_papers():
    """Upload test papers for evaluation"""
    try:
        print("DEBUG: Upload endpoint called")
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        test_type = request.form.get('test_type', 'Pre-Test')
        training_name = request.form.get('training_name', 'General Training')
        faculty_name = request.form.get('faculty_name', '')
        training_date = request.form.get('training_date')
        ticket_no = request.form.get('ticket_no', '')
        user_id = request.form.get('user_id', 1)  # Default user ID
        
        print(f"DEBUG: Received {len(files)} files, test_type={test_type}, training_name={training_name}, training_date={training_date}, user_id={user_id}")
        
        # Validate inputs
        if not test_type or not training_name:
            return jsonify({"error": "Test type and training name are required"}), 400
        
        # Create a new database session for this request
        db_session = get_db_session()
        
        try:
            # Validate that all files are Excel files
            for file in files:
                if file.filename == '':
                    continue
                filename = secure_filename(file.filename)
                if not (filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls')):
                    db_session.close()
                    return jsonify({"error": f"Only Excel files (.xlsx, .xls) are supported. Invalid file: {filename}"}), 400
            
            uploaded_files = []
            extracted_data = []
            tests = []  # Store test objects for bulk operations
            
            # First, create all test records
            for file in files:
                if file.filename == '':
                    continue
                    
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Ensure upload directory exists
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                
                try:
                    file.save(file_path)
                except Exception as e:
                    db_session.rollback()
                    db_session.close()
                    return jsonify({"error": f"Failed to save file {filename}: {str(e)}"}), 500
                
                # Create test record in database
                test = Test(
                    user_id=int(user_id),  # Ensure user_id is an integer
                    test_type=test_type,
                    training_name=training_name,
                    faculty_name=faculty_name,
                    date_conducted=None,  # Will be set after parsing training_date
                    file_path=file_path,
                    status='uploaded',
                    batch_ticket_no=ticket_no if ticket_no else None
                )                
                # Parse training_date if provided
                if training_date:
                    try:
                        # Handle different date formats
                        if 'T' in training_date:
                            # ISO format with time
                            test.date_conducted = datetime.fromisoformat(training_date)
                        else:
                            # Date only format
                            test.date_conducted = datetime.strptime(training_date, '%Y-%m-%d')
                    except ValueError:
                        # If parsing fails, use current time
                        test.date_conducted = datetime.utcnow()
                else:
                    # Default to current time if no date provided
                    test.date_conducted = datetime.utcnow()
                
                db_session.add(test)
                tests.append((test, filename, file_path))  # Store test with its metadata
            
            # Commit all test records at once
            db_session.commit()
            
            # Now process each file
            for test, filename, file_path in tests:
                try:
                    print(f"DEBUG: Processing file {filename}")
                    # Process Excel files only
                    if filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls'):
                        # For Excel files, process differently
                        print(f"DEBUG: Calling ocr_extractor.process_excel for {filename}")
                        extracted = ocr_extractor.process_excel(file_path, ticket_no=ticket_no)
                        print(f"DEBUG: Finished ocr_extractor.process_excel for {filename}")
                    else:
                        # This should not happen due to validation above, but just in case
                        extracted = {
                            'message': f"File type not supported for detailed processing: {filename}",
                            'file_name': filename,
                            'test_type': test_type
                        }
                    
                    # Handle NaN values in extracted data
                    def handle_nan_values(data):
                        if isinstance(data, dict):
                            return {k: handle_nan_values(v) for k, v in data.items()}
                        elif isinstance(data, list):
                            return [handle_nan_values(item) for item in data]
                        elif pd.isna(data):
                            return None
                        else:
                            return data
                    
                    # Clean the extracted data
                    cleaned_extracted = handle_nan_values(extracted)
                    
                    # Save extracted data
                    extracted_data.append({
                        'test_id': test.test_id,
                        'filename': filename,
                        'extracted': cleaned_extracted
                    })
                    
                    # Update test status
                    test.status = 'processing'
                    uploaded_files.append(filename)
                    
                    # For Excel files, automatically evaluate them
                    if filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls'):
                        try:
                            # Process the Excel file and save evaluations
                            # Handle different data structures from OCR extractor
                            candidates = []
                            if 'candidates' in cleaned_extracted:
                                candidates = cleaned_extracted.get('candidates', [])
                            elif 'sample_data' in cleaned_extracted:
                                # Convert sample_data to candidates format
                                sample_data = cleaned_extracted.get('sample_data', [])
                                for row in sample_data:
                                    candidate = {}
                                    # Extract marks from question columns
                                    marks = {}
                                    for key, value in row.items():
                                        if key.startswith('Question '):
                                            try:
                                                marks[key] = float(value) if value else 0
                                            except (ValueError, TypeError):
                                                marks[key] = 0
                                    
                                    candidate['marks'] = marks
                                    candidate['name'] = row.get('Name', '')
                                    candidate['ticket_no'] = row.get('Ticket No', row.get('Name', ''))
                                    candidates.append(candidate)
                            elif isinstance(cleaned_extracted, dict) and 'message' in cleaned_extracted and 'could not identify' in cleaned_extracted.get('message', ''):
                                # Handle unrecognized file types - try to process them anyway
                                print(f"Warning: Unrecognized file structure for {filename}, attempting to process sample data")
                                sample_data = cleaned_extracted.get('sample_data', [])
                                for row in sample_data:
                                    candidate = {}
                                    # Extract marks from question columns
                                    marks = {}
                                    for key, value in row.items():
                                        if key.startswith('Question '):
                                            try:
                                                marks[key] = float(value) if value else 0
                                            except (ValueError, TypeError):
                                                marks[key] = 0
                                    
                                    candidate['marks'] = marks
                                    candidate['name'] = row.get('Name', '')
                                    candidate['ticket_no'] = row.get('Ticket No', row.get('Name', ''))
                                    candidates.append(candidate)
                            
                            # Process each candidate
                            candidate_counter = 0  # Track candidate count for this test
                            for candidate in candidates:
                                # Create a unique candidate ID
                                candidate_id = candidate.get('ticket_no', candidate.get('name', f"candidate_{test.test_id}_{candidate_counter}"))
                                candidate_counter += 1
                                
                                # Check if this is a quiz_test (POSH training) or standard test
                                if 'total_points' in candidate and 'max_possible_points' in candidate:
                                    # POSH training data with points
                                    total_score = candidate['total_points']
                                    max_possible_score = candidate['max_possible_points']
                                    percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
                                else:
                                    # Standard test data
                                    total_score = sum(candidate['marks'].values())
                                    # For standard tests, assume total test is out of 20 marks (as per requirement)
                                    max_possible_score = 20
                                    percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
                                
                                # Determine pass/fail status (40% for all tests)
                                status = "pass" if percentage >= 40 else "fail"
                                
                                # Save evaluation to database
                                evaluation = Evaluation(
                                    test_id=test.test_id,
                                    candidate_id=candidate_id,
                                    total_score=total_score,
                                    percentage=percentage,
                                    status=status
                                )
                                db_session.add(evaluation)
                            
                            # Update test status to completed
                            test.status = 'completed'
                            
                            # Commit the evaluations for this test to avoid database locking issues
                            try:
                                db_session.commit()
                            except Exception as commit_error:
                                print(f"Warning: Could not commit evaluations for test {test.test_id}: {str(commit_error)}")
                                db_session.rollback()
                                test.status = 'failed'
                                db_session.commit()
                        except Exception as eval_error:
                            print(f"Warning: Failed to auto-evaluate test {test.test_id}: {str(eval_error)}")
                            import traceback
                            traceback.print_exc()
                            # Rollback any changes and mark test as failed
                            try:
                                db_session.rollback()
                                test.status = 'failed'
                                db_session.commit()
                            except Exception as rollback_error:
                                print(f"Warning: Could not rollback test {test.test_id}: {str(rollback_error)}")
                    
                except Exception as e:
                    test.status = 'failed'
                    db_session.commit()  # Commit the failed status
                    db_session.close()  # Close the session before returning
                    return jsonify({"error": f"Failed to process {filename}: {str(e)}"}), 500
            
            # Commit all status updates at once
            db_session.commit()
            
            # Handle NaN values in the final response
            def handle_nan_in_response(data):
                if isinstance(data, dict):
                    return {k: handle_nan_in_response(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [handle_nan_in_response(item) for item in data]
                elif pd.isna(data):
                    return None
                else:
                    return data
            
            # Send both summary and full data for proper display
            full_extracted_data = []
            for item in extracted_data:
                if item and isinstance(item, dict) and 'extracted' in item:
                    extracted_content = item['extracted']
                    if extracted_content and isinstance(extracted_content, dict):
                        # Create a response with both summary and full data
                        full_extracted = {
                            'test_id': item.get('test_id'),
                            'filename': item.get('filename'),
                            'extracted': {
                                'total_candidates': len(extracted_content.get('candidates', []) if 'candidates' in extracted_content else extracted_content.get('sample_data', [])),
                                'columns': extracted_content.get('columns', [])[:10],  # Limit columns
                                'message': f"Processed {len(extracted_content.get('candidates', []) if 'candidates' in extracted_content else extracted_content.get('sample_data', []))} candidates"
                            },
                            'full_data': extracted_content  # Include full data for display
                        }
                        
                        # No POSH-specific info needed
                        
                        # Add standard Excel info if present
                        if 'name_column' in extracted_content:
                            full_extracted['extracted']['name_column'] = extracted_content['name_column']
                        if 'ticket_column' in extracted_content:
                            full_extracted['extracted']['ticket_column'] = extracted_content['ticket_column']
                        if 'mark_columns' in extracted_content:
                            full_extracted['extracted']['mark_columns'] = extracted_content['mark_columns'][:5]  # Limit mark columns
                        if 'point_columns' in extracted_content:
                            full_extracted['extracted']['point_columns'] = extracted_content['point_columns'][:5]  # Limit point columns
                        if 'file_type' in extracted_content:
                            full_extracted['extracted']['file_type'] = extracted_content['file_type']
                            
                        full_extracted_data.append(full_extracted)
                    else:
                        full_extracted_data.append(item)
                else:
                    full_extracted_data.append(item)
            
            cleaned_extracted_data = handle_nan_in_response(full_extracted_data)
            cleaned_uploaded_files = handle_nan_in_response(uploaded_files)
            
            db_session.close()  # Close the session
            return jsonify({
                "message": f"Successfully uploaded {len(cleaned_uploaded_files)} files",
                "files": cleaned_uploaded_files,
                "extracted_data": cleaned_extracted_data
            }), 201
            
        except Exception as e:
            # Rollback in case of any error and close session
            print(f"DEBUG: Exception in upload: {str(e)}")
            import traceback
            traceback.print_exc()
            db_session.rollback()
            db_session.close()
            return jsonify({"error": f"Upload failed: {str(e)}"}), 500
    except Exception as e:
        print(f"DEBUG: Unexpected exception in upload endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/api/trainings', methods=['GET'])
def get_trainings():
    """Get all unique training names"""
    # Create a new database session for this request
    db_session = get_db_session()
    try:
        # Get distinct training names from the Test table
        trainings = db_session.query(Test.training_name).distinct().all()
        result = [training[0] for training in trainings]
        db_session.close()
        return jsonify(result)
    except Exception as e:
        import traceback
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        db_session.close()
        return jsonify(error_details), 500

@app.route('/api/faculties', methods=['GET'])
def get_faculties():
    """Get all unique faculty names"""
    # Create a new database session for this request
    db_session = get_db_session()
    try:
        # Get distinct faculty names from the Test table
        faculties = db_session.query(Test.faculty_name).distinct().all()
        result = [faculty[0] for faculty in faculties if faculty[0] is not None]
        db_session.close()
        return jsonify(result)
    except Exception as e:
        import traceback
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        db_session.close()
        return jsonify(error_details), 500
@app.route('/api/tests', methods=['GET'])
def get_tests():
    """Get all tests"""
    # Create a new database session for this request
    db_session = get_db_session()
    try:
        tests = db_session.query(Test).all()
        result = [serialize_object(test) for test in tests]
        db_session.close()
        return jsonify(result)
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/tests/<int:test_id>', methods=['GET'])
def get_test(test_id):
    """Get specific test details"""
    # Create a new database session for this request
    db_session = get_db_session()
    try:
        test = db_session.query(Test).filter(Test.test_id == test_id).first()
        if not test:
            db_session.close()
            return jsonify({"error": "Test not found"}), 404
        result = serialize_object(test)
        db_session.close()
        return jsonify(result)
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

# Manual evaluation endpoint removed - all evaluations are now automatic
    # @app.route('/api/tests/<int:test_id>/evaluate', methods=['POST'])
    # def evaluate_test(test_id):
    #     """Evaluate a test with provided answer key"""
    #     # This endpoint has been removed as all evaluations are now automatic
    #     return jsonify({"error": "Manual evaluation is no longer supported. All evaluations are automatic."}), 400

@app.route('/api/analytics/batch/<int:test_id>', methods=['GET'])
def get_batch_analytics(test_id):
    """Get batch analytics for a test"""
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        test = db_session.query(Test).filter(Test.test_id == test_id).first()
        if not test:
            db_session.close()
            return jsonify({"error": "Test not found"}), 404
        
        # Retrieve all evaluations for this test
        evaluations = db_session.query(Evaluation).filter(Evaluation.test_id == test_id).all()
        
        if not evaluations:
            # If no evaluations exist, try to get data from the file directly
            if test.file_path and os.path.exists(test.file_path):
                try:
                    if test.file_path.endswith('.xlsx') or test.file_path.endswith('.xls'):
                        # Process Excel file directly
                        excel_data = ocr_extractor.process_excel(test.file_path)
                        # Handle different data structures from OCR extractor
                        candidates = []
                        if 'candidates' in excel_data:
                            candidates = excel_data.get('candidates', [])
                        elif 'sample_data' in excel_data:
                            # Convert sample_data to candidates format
                            sample_data = excel_data.get('sample_data', [])
                            for row in sample_data:
                                candidate = {}
                                # Extract marks from question columns
                                marks = {}
                                for key, value in row.items():
                                    if key.startswith('Question '):
                                        try:
                                            marks[key] = float(value) if value else 0
                                        except (ValueError, TypeError):
                                            marks[key] = 0
                                
                                candidate['marks'] = marks
                                candidate['name'] = row.get('Name', '')
                                candidate['ticket_no'] = row.get('Ticket No', row.get('Name', ''))
                                candidates.append(candidate)
                        
                        # Calculate statistics for all test files uniformly
                        scores = []
                        for candidate in candidates:
                            # Check if this is a quiz_test (POSH training) or standard test
                            if 'total_points' in candidate and 'max_possible_points' in candidate:
                                # POSH training data with points
                                total_score = candidate['total_points']
                                max_possible_score = candidate['max_possible_points']
                                percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
                            else:
                                # Standard test data
                                total_score = sum(candidate['marks'].values())
                                # For standard tests, assume total test is out of 20 marks (as per requirement)
                                max_possible_score = 20
                                percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
                            scores.append(percentage)
                        
                        if scores:
                            batch_stats = {
                                "average_score": statistics.mean(scores),
                                "median_score": statistics.median(scores),
                                "highest_score": max(scores),
                                "lowest_score": min(scores),
                                "pass_rate": (len([s for s in scores if s >= 40]) / len(scores)) * 100 if scores and len(scores) > 0 else 0,
                                "total_candidates": len(scores)
                            }
                            
                            # Additional analytics for better insights
                            # Performance distribution
                            percentages = scores
                            excellent = len([p for p in percentages if p >= 90])
                            good = len([p for p in percentages if 70 <= p < 90])
                            average = len([p for p in percentages if 50 <= p < 70])
                            poor = len([p for p in percentages if p < 50])
                            
                            performance_distribution = {
                                "excellent": excellent,
                                "good": good,
                                "average": average,
                                "poor": poor
                            }
                            
                            # Weak areas analysis for all test files
                            weak_areas = []
                            if 'candidates' in excel_data and excel_data['candidates']:
                                # Analyze question-wise performance
                                question_scores = {}
                                for candidate in excel_data['candidates']:
                                    # Get marks based on file type
                                    marks_data = candidate.get('marks', {})
                                    for question, score in marks_data.items():
                                        if question not in question_scores:
                                            question_scores[question] = []
                                        question_scores[question].append(score)
                                
                                # Calculate average score per question
                                question_averages = {}
                                for question, scores_list in question_scores.items():
                                    if scores_list and len(scores_list) > 0:
                                        question_averages[question] = sum(scores_list) / len(scores_list)
                                
                                # Sort questions by average score (ascending) to find weak areas
                                sorted_questions = sorted(question_averages.items(), key=lambda x: x[1])
                                weak_areas = [q[0] for q in sorted_questions[:3]]  # Top 3 weak areas
                            
                            result = jsonify({
                                "test_id": test_id,
                                "test_type": test.test_type,
                                "training_name": test.training_name,
                                "batch_stats": batch_stats,
                                "performance_distribution": performance_distribution,
                                "weak_areas": weak_areas if weak_areas else ["Not enough data for analysis"],
                                "file_type": "regular",
                                "total_questions": 0
                            })
                            db_session.close()
                            return result
                except Exception as e:
                    db_session.close()
                    return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
            
            result = jsonify({
                "test_id": test_id,
                "test_type": test.test_type,
                "training_name": test.training_name,
                "batch_stats": {
                    "average_score": 0,
                    "median_score": 0,
                    "highest_score": 0,
                    "lowest_score": 0,
                    "pass_rate": 0,
                    "total_candidates": 0
                },
                "message": "No evaluations found for this test"
            })
            db_session.close()
            return result
        
        # Calculate batch statistics from evaluations
        scores = [eval.total_score for eval in evaluations]
        percentages = [eval.percentage for eval in evaluations]
        
        batch_stats = {
            "average_score": statistics.mean(scores) if scores else 0,
            "median_score": statistics.median(scores) if scores else 0,
            "highest_score": max(scores) if scores else 0,
            "lowest_score": min(scores) if scores else 0,
            "pass_rate": (len([s for s in percentages if s >= 40]) / len(percentages)) * 100 if percentages and len(percentages) > 0 else 0,
            "total_candidates": len(scores),
            "average_percentage": statistics.mean(percentages) if percentages else 0
        }
        
        # Additional analytics for better insights
        # Performance distribution
        excellent = len([p for p in percentages if p >= 90])
        good = len([p for p in percentages if 70 <= p < 90])
        average = len([p for p in percentages if 50 <= p < 70])
        poor = len([p for p in percentages if p < 50])
        
        performance_distribution = {
            "excellent": excellent,
            "good": good,
            "average": average,
            "poor": poor
        }
        
        # Top performers (sort by percentage and take top 5)
        top_performers = []
        sorted_evaluations = sorted(evaluations, key=lambda x: x.percentage, reverse=True)
        for eval_record in sorted_evaluations[:5]:  # Top 5 performers
            top_performers.append({
                "name": eval_record.candidate_id,
                "ticket_no": eval_record.candidate_id,
                "average_score": eval_record.percentage
            })
        
        # Weak areas analysis (simplified approach)
        # In a real implementation, this would analyze question-by-question performance
        weak_areas = []
        if percentages:
            avg_percentage = statistics.mean(percentages)
            # If average is below 70%, we consider general areas for improvement
            if avg_percentage < 70:
                weak_areas = [
                    {"subject": "Overall Knowledge", "average_score": avg_percentage},
                    {"subject": "Core Concepts", "average_score": avg_percentage * 0.9},
                    {"subject": "Application Skills", "average_score": avg_percentage * 0.85}
                ]
            else:
                weak_areas = [
                    {"subject": "Advanced Topics", "average_score": avg_percentage * 0.8},
                    {"subject": "Practical Application", "average_score": avg_percentage * 0.85},
                    {"subject": "Specialized Skills", "average_score": avg_percentage * 0.9}
                ]
        
        result = jsonify({
            "test_id": test_id,
            "test_type": test.test_type,
            "training_name": test.training_name,
            "batch_stats": batch_stats,
            "performance_distribution": performance_distribution,
            "top_performers": top_performers,
            "weak_areas": weak_areas
        })
        db_session.close()
        return result
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500
@app.route('/api/analytics/candidate/<candidate_id>', methods=['GET'])
def get_candidate_analytics(candidate_id):
    """Get candidate analytics"""
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        # Get all evaluations for this candidate
        evaluations = db_session.query(Evaluation).filter(Evaluation.candidate_id == candidate_id).all()
        
        if not evaluations:
            db_session.close()
            return jsonify({"error": "No evaluations found for this candidate"}), 404
        
        # Calculate improvement percentage (comparing first and last test)
        improvement_percentage = 0
        if len(evaluations) > 1:
            first_score = evaluations[0].percentage
            last_score = evaluations[-1].percentage
            if first_score > 0:
                improvement_percentage = ((last_score - first_score) / first_score) * 100
        
        # Prepare performance history
        performance_history = []
        for eval_record in evaluations:
            # Get test details
            test = db_session.query(Test).filter(Test.test_id == eval_record.test_id).first()
            if test:
                performance_history.append({
                    'test_id': test.test_id,
                    'test_type': test.test_type,
                    'training_name': test.training_name,
                    'score': eval_record.total_score,
                    'percentage': eval_record.percentage,
                    'status': eval_record.status,
                    'date': test.date_uploaded.isoformat() if test.date_uploaded else None
                })
        
        result = jsonify({
            "candidate_id": candidate_id,
            "improvement_percentage": improvement_percentage,
            "performance_history": performance_history,
            "total_evaluations": len(evaluations)
        })
        db_session.close()
        return result
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/overall', methods=['GET'])
def get_overall_analytics():
    """Get overall analytics across all tests"""
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        # Retrieve all evaluations across all tests
        evaluations = db_session.query(Evaluation).all()
        
        if not evaluations:
            # If no evaluations exist, try to get data from all test files directly
            tests = db_session.query(Test).all()
            all_scores = []
            all_percentages = []
            total_candidates = 0
            passed_candidates = 0
            
            for test in tests:
                if test.file_path and os.path.exists(test.file_path):
                    try:
                        if test.file_path.endswith('.xlsx') or test.file_path.endswith('.xls'):
                            # Process Excel file directly
                            excel_data = ocr_extractor.process_excel(test.file_path)
                            candidates = excel_data['candidates']
                            
                            # Calculate statistics uniformly for all test files
                            scores = []
                            for candidate in candidates:
                                # Calculate total score from all marks
                                total_score = sum(candidate['marks'].values())
                                scores.append(total_score)
                            
                            all_scores.extend(scores)
                            # Calculate percentages for these scores (uniformly for all test files)
                            for i, score in enumerate(scores):
                                # For all tests, assuming total test is out of 20 marks (as per requirement)
                                max_possible = 20
                                percentage = (score / max_possible) * 100 if max_possible > 0 else 0
                                all_percentages.append(percentage)
                                
                                # Count passed candidates (uniform threshold for all tests)
                                if percentage >= 40:
                                    passed_candidates += 1
                            
                            total_candidates += len(scores)
                    except Exception as e:
                        # Log the error but continue processing other files
                        print(f"Warning: Failed to process file {test.file_path if 'test' in locals() else 'unknown'}: {str(e)}")
                        continue  # Skip files that can't be processed
            
            if all_scores:
                overall_stats = {
                    "average_score": statistics.mean(all_scores),
                    "median_score": statistics.median(all_scores),
                    "highest_score": max(all_scores),
                    "lowest_score": min(all_scores),
                    "pass_rate": (passed_candidates / total_candidates) * 100 if total_candidates > 0 else 0,
                    "total_candidates": total_candidates,
                    "passed_candidates": passed_candidates,
                    "failed_candidates": total_candidates - passed_candidates,
                    "total_tests": len(tests)
                }
                
                # Performance distribution
                excellent = len([p for p in all_percentages if p >= 90])
                good = len([p for p in all_percentages if 70 <= p < 90])
                average = len([p for p in all_percentages if 50 <= p < 70])
                poor = len([p for p in all_percentages if p < 50])
                
                performance_distribution = {
                    "excellent": excellent,
                    "good": good,
                    "average": average,
                    "poor": poor
                }
                
                result = jsonify({
                    "overall_stats": overall_stats,
                    "performance_distribution": performance_distribution
                })
                db_session.close()
                return result
        
        # Calculate overall statistics from evaluations
        scores = [eval.total_score for eval in evaluations]
        percentages = [eval.percentage for eval in evaluations]
        
        # Count passed candidates
        passed_candidates = len([s for s in percentages if s >= 40])
        total_candidates = len(percentages)
        
        overall_stats = {
            "average_score": statistics.mean(scores) if scores else 0,
            "median_score": statistics.median(scores) if scores else 0,
            "highest_score": max(scores) if scores else 0,
            "lowest_score": min(scores) if scores else 0,
            "pass_rate": (passed_candidates / total_candidates) * 100 if total_candidates > 0 else 0,
            "total_candidates": total_candidates,
            "passed_candidates": passed_candidates,
            "failed_candidates": total_candidates - passed_candidates,
            "total_tests": db_session.query(Test).count()
        }
        
        # Performance distribution
        excellent = len([p for p in percentages if p >= 90])
        good = len([p for p in percentages if 70 <= p < 90])
        average = len([p for p in percentages if 50 <= p < 70])
        poor = len([p for p in percentages if p < 50])
        
        performance_distribution = {
            "excellent": excellent,
            "good": good,
            "average": average,
            "poor": poor
        }
        
        result = jsonify({
            "overall_stats": overall_stats,
            "performance_distribution": performance_distribution
        })
        db_session.close()
        return result
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/combined-pass-rate', methods=['GET'])
def get_combined_pass_rate():
    """Get combined pass rate across all trainings in Excel sheets"""
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        # Try to get data from all test files directly
        tests = db_session.query(Test).all()
        all_candidates = []
        training_stats = []
        
        # Process each test file
        for test in tests:
            if test.file_path and os.path.exists(test.file_path):
                try:
                    if test.file_path.endswith('.xlsx') or test.file_path.endswith('.xls'):
                        # Process Excel file directly
                        excel_data = ocr_extractor.process_excel(test.file_path)
                        candidates = excel_data.get('candidates', [])
                        
                        # Add training info
                        training_info = {
                            "test_id": test.test_id,
                            "training_name": test.training_name,
                            "test_type": test.test_type,
                            "total_candidates": len(candidates),
                            # All files are treated uniformly
                        }
                        
                        # Calculate pass rate for this training
                        passed_count = 0
                        for candidate in candidates:
                            # Uniform pass threshold for all tests
                            percentage = candidate.get('percentage', 0)
                            if percentage >= 40:
                                passed_count += 1
                        
                        training_info["passed_candidates"] = passed_count
                        training_info["failed_candidates"] = len(candidates) - passed_count
                        training_info["pass_rate"] = (passed_count / len(candidates)) * 100 if len(candidates) > 0 else 0
                        
                        training_stats.append(training_info)
                        
                        # Add candidates to overall list
                        for candidate in candidates:
                            candidate_data = candidate.copy()
                            candidate_data["training_name"] = test.training_name
                            candidate_data["test_type"] = test.test_type
                            # All files are treated uniformly
                            all_candidates.append(candidate_data)
                            
                except Exception as e:
                    # Log the error but continue processing other files
                    print(f"Warning: Failed to process file {test.file_path}: {str(e)}")
                    continue  # Skip files that can't be processed
        
        # Calculate combined pass rate
        total_candidates = len(all_candidates)
        passed_candidates = 0
        
        for candidate in all_candidates:
            percentage = candidate.get("percentage", 0)
            
            # Apply uniform pass threshold for all tests
            if percentage >= 40:
                passed_candidates += 1
        
        combined_pass_rate = (passed_candidates / total_candidates) * 100 if total_candidates > 0 else 0
        
        # Group by training name for summary
        training_summary = {}
        for training in training_stats:
            name = training["training_name"]
            if name not in training_summary:
                training_summary[name] = {
                    "training_name": name,
                    "total_trainings": 0,
                    "total_candidates": 0,
                    "passed_candidates": 0,
                    "failed_candidates": 0,
                    "combined_pass_rate": 0
                }
            
            training_summary[name]["total_trainings"] += 1
            training_summary[name]["total_candidates"] += training["total_candidates"]
            training_summary[name]["passed_candidates"] += training["passed_candidates"]
            training_summary[name]["failed_candidates"] += training["failed_candidates"]
            
        # Calculate pass rates for each training
        for name in training_summary:
            total = training_summary[name]["total_candidates"]
            passed = training_summary[name]["passed_candidates"]
            training_summary[name]["combined_pass_rate"] = (passed / total) * 100 if total > 0 else 0
        
        result = jsonify({
            "combined_statistics": {
                "total_candidates": total_candidates,
                "passed_candidates": passed_candidates,
                "failed_candidates": total_candidates - passed_candidates,
                "combined_pass_rate": combined_pass_rate
            },
            "training_summary": list(training_summary.values()),
            "individual_trainings": training_stats
        })
        
        db_session.close()
        return result
        
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/learning-index', methods=['GET'])
def get_overall_learning_index():
    """Get overall learning index across all pre-post test comparisons"""
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        # Get all pre-test and post-test pairs
        # We'll look for tests that have the same training name and faculty name
        # and where one is Pre-Test and the other is Post-Test
        tests = db_session.query(Test).all()
        
        # Group tests by training_name and faculty_name
        test_groups = {}
        for test in tests:
            key = (test.training_name, test.faculty_name)
            if key not in test_groups:
                test_groups[key] = {'pre_tests': [], 'post_tests': []}
            
            if test.test_type == 'Pre-Test':
                test_groups[key]['pre_tests'].append(test)
            elif test.test_type == 'Post-Test':
                test_groups[key]['post_tests'].append(test)
        
        # Calculate learning indices for each group
        learning_indices = []
        for key, group in test_groups.items():
            # For each pre-test, find the corresponding post-test
            for pre_test in group['pre_tests']:
                for post_test in group['post_tests']:
                    # Compare these tests
                    try:
                        # Call the comparison endpoint internally
                        from flask import current_app
                        with current_app.test_client() as client:
                            response = client.post('/api/compare/tests', 
                                                 json={
                                                     'pre_test_id': pre_test.test_id,
                                                     'post_test_id': post_test.test_id
                                                 })
                            
                            if response.status_code == 200:
                                comparison_data = response.get_json()
                                if 'comparison' in comparison_data and 'statistics' in comparison_data['comparison']:
                                    learning_index = comparison_data['comparison']['statistics'].get('learning_index', 0)
                                    learning_indices.append(learning_index)
                    except Exception as e:
                        print(f"Warning: Failed to compare tests {pre_test.test_id} and {post_test.test_id}: {str(e)}")
                        continue
        
        # Calculate overall learning index
        overall_learning_index = 0
        if learning_indices:
            overall_learning_index = sum(learning_indices) / len(learning_indices)
        
        result = jsonify({
            "overall_learning_index": round(overall_learning_index, 2),
            "total_comparisons": len(learning_indices),
            "individual_indices": learning_indices
        })
        
        db_session.close()
        return result
        
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/batch/<int:test_id>', methods=['GET'])
def generate_batch_report(test_id):
    """Generate batch report"""
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        test = db_session.query(Test).filter(Test.test_id == test_id).first()
        if not test:
            db_session.close()
            return jsonify({"error": "Test not found"}), 404
        
        # Get all evaluations for this test
        evaluations = db_session.query(Evaluation).filter(Evaluation.test_id == test_id).all()
        
        if not evaluations:
            db_session.close()
            return jsonify({"error": "No evaluations found for this test"}), 404
        
        # Calculate statistics
        scores = [eval.total_score for eval in evaluations]
        percentages = [eval.percentage for eval in evaluations]
        total_candidates = len(evaluations)
        average_score = sum(scores) / len(scores) if scores else 0
        average_percentage = sum(percentages) / len(percentages) if percentages else 0
        highest_score = max(scores) if scores else 0
        lowest_score = min(scores) if scores else 0
        pass_count = len([p for p in percentages if p >= 40])  # Assuming 40% is pass mark
        pass_rate = (pass_count / total_candidates * 100) if total_candidates > 0 else 0
        
        # Calculate learning index if this is a post-test
        learning_index = 0
        pre_test_evaluations = []
        if test.test_type.lower() == 'post-test':
            # Look for corresponding pre-test (same training name, different type)
            pre_test = db_session.query(Test).filter(
                Test.training_name == test.training_name,
                Test.test_type == 'Pre-Test'
            ).first()
            
            if pre_test:
                pre_test_evaluations = db_session.query(Evaluation).filter(
                    Evaluation.test_id == pre_test.test_id
                ).all()
                
                # Calculate learning index based on improvement
                if pre_test_evaluations:
                    post_scores = {eval.candidate_id: eval.percentage for eval in evaluations}
                    pre_scores = {eval.candidate_id: eval.percentage for eval in pre_test_evaluations}
                    
                    common_candidates = set(post_scores.keys()) & set(pre_scores.keys())
                    if common_candidates:
                        improvements = []
                        for candidate_id in common_candidates:
                            pre_score = pre_scores[candidate_id]
                            post_score = post_scores[candidate_id]
                            improvement = ((post_score - pre_score) / pre_score * 100) if pre_score > 0 else 0
                            improvements.append(improvement)
                        learning_index = sum(improvements) / len(improvements) if improvements else 0
        
        # Get top performers
        sorted_evaluations = sorted(evaluations, key=lambda x: x.percentage, reverse=True)
        top_performers = []
        for eval in sorted_evaluations[:3]:  # Top 3 performers
            top_performers.append({
                "candidate_id": eval.candidate_id,
                "score": eval.total_score,
                "percentage": eval.percentage,
                "status": eval.status
            })
        
        # Performance distribution
        excellent_count = len([p for p in percentages if p >= 80])
        good_count = len([p for p in percentages if 60 <= p < 80])
        average_count = len([p for p in percentages if 40 <= p < 60])
        poor_count = len([p for p in percentages if p < 40])
        
        # Prepare report data
        report_data = {
            "report_type": "Batch Report",
            "test_id": test_id,
            "test_type": test.test_type,
            "training_name": test.training_name,
            "faculty_name": test.faculty_name,
            "date_conducted": test.date_conducted.isoformat() if test.date_conducted else None,
            "generated_at": datetime.utcnow().isoformat(),
            "batch_statistics": {
                "total_candidates": total_candidates,
                "average_score": round(average_score, 2),
                "average_percentage": round(average_percentage, 2),
                "pass_rate": round(pass_rate, 2),
                "highest_score": highest_score,
                "lowest_score": lowest_score,
                "learning_index": round(learning_index, 2),
                "performance_distribution": {
                    "excellent": excellent_count,
                    "good": good_count,
                    "average": average_count,
                    "poor": poor_count
                }
            },
            "top_performers": top_performers,
            "total_candidates": total_candidates,
            "passed_candidates": pass_count,
            "failed_candidates": total_candidates - pass_count
        }
        
        # Check if PDF format is requested
        format_type = request.args.get('format', 'json')
        if format_type == 'pdf':
            from io import BytesIO
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            
            # Create a BytesIO buffer to hold the PDF
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            elements = []
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            
            # Add title
            title = Paragraph(f"Batch Performance Report - {test.training_name} ({test.test_type})", title_style)
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            # Add test details
            details_text = f"""
            <b>Test ID:</b> {test_id}<br/>
            <b>Test Type:</b> {test.test_type}<br/>
            <b>Training Name:</b> {test.training_name}<br/>
            <b>Faculty:</b> {test.faculty_name or 'N/A'}<br/>
            <b>Date Conducted:</b> {test.date_conducted.strftime('%Y-%m-%d') if test.date_conducted else 'N/A'}<br/>
            <b>Generated On:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            """
            elements.append(Paragraph(details_text, styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Add overall statistics
            elements.append(Paragraph("Overall Statistics", heading_style))
            
            stats_data = [
                ["Metric", "Value"],
                ["Total Candidates", str(report_data['total_candidates'])],
                ["Passed", str(report_data['passed_candidates'])],
                ["Failed", str(report_data['failed_candidates'])],
                ["Pass Rate", f"{report_data['batch_statistics']['pass_rate']}%"],
                ["Average Score", str(report_data['batch_statistics']['average_score'])],
                ["Average Percentage", f"{report_data['batch_statistics']['average_percentage']}%"],
                ["Highest Score", str(report_data['batch_statistics']['highest_score'])],
                ["Lowest Score", str(report_data['batch_statistics']['lowest_score'])],
                ["Learning Index", f"{report_data['batch_statistics']['learning_index']}%"],
            ]
            
            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(stats_table)
            elements.append(Spacer(1, 20))
            
            # Add performance distribution
            elements.append(Paragraph("Performance Distribution", heading_style))
            
            dist_data = [
                ["Performance Level", "Count", "Percentage"],
                ["Excellent (≥80%)", str(report_data['batch_statistics']['performance_distribution']['excellent']), 
                 f"{report_data['batch_statistics']['performance_distribution']['excellent']/total_candidates*100:.1f}%"],
                ["Good (60-79%)", str(report_data['batch_statistics']['performance_distribution']['good']), 
                 f"{report_data['batch_statistics']['performance_distribution']['good']/total_candidates*100:.1f}%"],
                ["Average (40-59%)", str(report_data['batch_statistics']['performance_distribution']['average']), 
                 f"{report_data['batch_statistics']['performance_distribution']['average']/total_candidates*100:.1f}%"],
                ["Poor (<40%)", str(report_data['batch_statistics']['performance_distribution']['poor']), 
                 f"{report_data['batch_statistics']['performance_distribution']['poor']/total_candidates*100:.1f}%"],
            ]
            
            dist_table = Table(dist_data)
            dist_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(dist_table)
            elements.append(Spacer(1, 20))
            
            # Add top performers
            if report_data['top_performers']:
                elements.append(Paragraph("Top Performers", heading_style))
                
                top_data = [["Rank", "Candidate ID", "Score", "Percentage", "Status"]]
                for i, performer in enumerate(report_data['top_performers'], 1):
                    top_data.append([
                        str(i), 
                        performer['candidate_id'], 
                        str(performer['score']), 
                        f"{performer['percentage']}%", 
                        performer['status'].title()
                    ])
                
                top_table = Table(top_data)
                top_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(top_table)
            
            # Build PDF
            doc.build(elements)
            buffer.seek(0)
            
            # Return PDF as download
            filename = f"batch_report_{test.training_name}_{test_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
            return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')
        
        result = jsonify(report_data)
        db_session.close()
        return result
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/candidate/<candidate_id>', methods=['GET'])
def generate_candidate_report(candidate_id):
    """Generate individual candidate report"""
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        # Get all evaluations for this candidate
        evaluations = db_session.query(Evaluation).filter(Evaluation.candidate_id == candidate_id).all()
        
        if not evaluations:
            db_session.close()
            return jsonify({"error": "No evaluations found for this candidate"}), 404
        
        # Get candidate's pre-test and post-test scores if available
        pre_test_eval = None
        post_test_eval = None
        
        for eval in evaluations:
            test = db_session.query(Test).filter(Test.test_id == eval.test_id).first()
            if test:
                if test.test_type.lower() == 'pre-test':
                    pre_test_eval = eval
                elif test.test_type.lower() == 'post-test':
                    post_test_eval = eval
        
        # Calculate improvement
        improvement_percentage = 0
        performance_trend = "No Change"
        if pre_test_eval and post_test_eval:
            pre_percentage = pre_test_eval.percentage
            post_percentage = post_test_eval.percentage
            if pre_percentage > 0:
                improvement_percentage = ((post_percentage - pre_percentage) / pre_percentage) * 100
            else:
                improvement_percentage = post_percentage if post_percentage > 0 else 0
            
            if improvement_percentage > 20:
                performance_trend = "Significant Improvement"
            elif improvement_percentage > 0:
                performance_trend = "Moderate Improvement"
            elif improvement_percentage == 0:
                performance_trend = "No Change"
            else:
                performance_trend = "Decline"
        
        # Get all tests for this candidate to analyze performance across all tests
        all_test_scores = []
        for eval in evaluations:
            test = db_session.query(Test).filter(Test.test_id == eval.test_id).first()
            if test:
                all_test_scores.append({
                    'test_id': test.test_id,
                    'test_type': test.test_type,
                    'training_name': test.training_name,
                    'score': eval.total_score,
                    'percentage': eval.percentage,
                    'status': eval.status,
                    'date_conducted': test.date_conducted.strftime('%Y-%m-%d') if test.date_conducted else None
                })
        
        # Prepare report data
        report_data = {
            "report_type": "Individual Report",
            "candidate_id": candidate_id,
            "generated_at": datetime.utcnow().isoformat(),
            "performance_summary": {
                "pre_test_score": pre_test_eval.percentage if pre_test_eval else 0,
                "post_test_score": post_test_eval.percentage if post_test_eval else 0,
                "improvement_percentage": round(improvement_percentage, 2),
                "performance_trend": performance_trend,
                "learning_index": round(improvement_percentage, 2) if improvement_percentage != 0 else 0
            },
            "all_test_scores": all_test_scores,
            "overall_status": "pass" if all(eval.status == "pass" for eval in evaluations) else "fail"
        }
        
        # If PDF format requested, generate PDF
        format_type = request.args.get('format', 'json')
        if format_type == 'pdf':
            from io import BytesIO
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            
            # Create a BytesIO buffer to hold the PDF
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            elements = []
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            
            # Add title
            title = Paragraph(f"Candidate Performance Report - {candidate_id}", title_style)
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            # Add candidate details
            details_text = f"""
            <b>Candidate ID:</b> {candidate_id}<br/>
            <b>Generated On:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Overall Status:</b> {report_data['overall_status'].title()}<br/>
            """
            elements.append(Paragraph(details_text, styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Add performance summary
            elements.append(Paragraph("Performance Summary", heading_style))
            
            summary_data = [
                ["Metric", "Value"],
                ["Pre-Test Score", f"{report_data['performance_summary']['pre_test_score']}%"],
                ["Post-Test Score", f"{report_data['performance_summary']['post_test_score']}%"],
                ["Improvement", f"{report_data['performance_summary']['improvement_percentage']}%"],
                ["Learning Index", f"{report_data['performance_summary']['learning_index']}%"],
                ["Performance Trend", report_data['performance_summary']['performance_trend']],
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
            
            # Add all test scores
            if report_data['all_test_scores']:
                elements.append(Paragraph("Test Performance History", heading_style))
                
                test_data = [["Test ID", "Test Type", "Training Name", "Score", "Percentage", "Status", "Date"]]
                for test_score in report_data['all_test_scores']:
                    test_data.append([
                        str(test_score['test_id']),
                        test_score['test_type'],
                        test_score['training_name'],
                        str(test_score['score']),
                        f"{test_score['percentage']}%",
                        test_score['status'].title(),
                        test_score['date_conducted'] or 'N/A'
                    ])
                
                test_table = Table(test_data)
                test_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(test_table)
            
            # Build PDF
            doc.build(elements)
            buffer.seek(0)
            
            # Return PDF as download
            filename = f"candidate_report_{candidate_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
            return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')
        
        result = jsonify(report_data)
        db_session.close()
        return result
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

# OneDrive integration routes
@app.route('/api/onedrive/attendance', methods=['GET'])
def get_attendance_from_onedrive():
    """Fetch attendance data from OneDrive"""
    # In a real implementation, you would:
    # 1. Connect to OneDrive API
    # 2. Fetch attendance sheets
    # 3. Parse and return data
    
    # For now, we'll return simulated data
    return jsonify({
        "message": "In a full implementation, this would fetch attendance data from OneDrive",
        "sample_data": [
            {"employee_id": "T00123", "punch_in": "2025-12-01T08:55:00", "punch_out": "2025-12-01T17:30:00"},
            {"employee_id": "T00456", "punch_in": "2025-12-01T09:02:00", "punch_out": "2025-12-01T17:25:00"}
        ]
    })

@app.route('/api/onedrive/documents', methods=['GET'])
def get_training_documents_from_onedrive():
    """Fetch training documents from OneDrive"""
    # In a real implementation, you would:
    # 1. Connect to OneDrive API
    # 2. Fetch training documents
    # 3. Return document metadata
    
    # For now, we'll return simulated data
    return jsonify({
        "message": "In a full implementation, this would fetch training documents from OneDrive",
        "sample_documents": [
            {"name": "Engine Basics.pdf", "size": "2.4MB", "last_modified": "2025-11-20"},
            {"name": "Electrical Systems Manual.pdf", "size": "5.1MB", "last_modified": "2025-11-22"}
        ]
    })

@app.route('/api/delete-all-data', methods=['DELETE'])
def delete_all_system_data():
    """Delete all Excel sheet data and uploaded files"""
    try:
        # Create a new database session for this request
        db_session = get_db_session()
        
        try:
            # Delete all records from tables in the correct order (due to foreign keys)
            db_session.query(Feedback).delete()
            db_session.query(Attendance).delete()
            db_session.query(Evaluation).delete()
            db_session.query(Answer).delete()
            db_session.query(Test).delete()
            # Note: We're not deleting User records as they might be needed for the system
            
            # Commit the changes
            db_session.commit()
            
            # Remove all files from the upload directory
            upload_folder = app.config['UPLOAD_FOLDER']
            
            # Check if upload folder exists
            if os.path.exists(upload_folder):
                # Remove all files in the upload directory
                for filename in os.listdir(upload_folder):
                    file_path = os.path.join(upload_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            import shutil
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Warning: Failed to delete {file_path}. Reason: {e}")
            
            return jsonify({
                "message": "All Excel sheet data and uploaded files have been successfully deleted!"
            }), 200
            
        except Exception as e:
            # Rollback in case of any error
            db_session.rollback()
            raise e
            
        finally:
            # Close the session
            db_session.close()
            
    except Exception as e:
        return jsonify({
            "error": f"Failed to delete data: {str(e)}"
        }), 500


@app.route('/api/compare/tests', methods=['POST'])
def compare_pre_post_tests():
    """Compare Pre-Test and Post-Test results to measure training impact"""
    try:
        data = request.get_json()
        pre_test_id = data.get('pre_test_id')
        post_test_id = data.get('post_test_id')
        
        if not pre_test_id or not post_test_id:
            return jsonify({"error": "Both pre_test_id and post_test_id are required"}), 400
        
        # Create database sessions
        db_session = get_db_session()
        
        # Get both tests
        pre_test = db_session.query(Test).filter(Test.test_id == pre_test_id).first()
        post_test = db_session.query(Test).filter(Test.test_id == post_test_id).first()
        
        if not pre_test or not post_test:
            db_session.close()
            return jsonify({"error": "One or both tests not found"}), 404
        
        # Get evaluations for both tests
        pre_evaluations = db_session.query(Evaluation).filter(Evaluation.test_id == pre_test_id).all()
        post_evaluations = db_session.query(Evaluation).filter(Evaluation.test_id == post_test_id).all()
        
        # If no evaluations exist, try to get data from the files directly
        if not pre_evaluations or not post_evaluations:
            # Process files directly if they exist
            pre_data = None
            post_data = None
            
            if pre_test.file_path and os.path.exists(pre_test.file_path):
                try:
                    pre_data = ocr_extractor.process_excel(pre_test.file_path)
                except Exception as e:
                    pass
            
            if post_test.file_path and os.path.exists(post_test.file_path):
                try:
                    post_data = ocr_extractor.process_excel(post_test.file_path)
                except Exception as e:
                    pass
            
            if pre_data and post_data:
                # Process data directly from files
                pre_candidates = pre_data['candidates']
                post_candidates = post_data['candidates']
                
                # Create dictionaries for easier comparison
                pre_scores = {}
                post_scores = {}
                
                # Collect all candidate IDs (uniformly for all test files)
                pre_candidate_ids = set()
                post_candidate_ids = set()
                
                for candidate in pre_candidates:
                    # Use ticket_no or name for candidate identification
                    candidate_id = candidate.get('ticket_no', candidate.get('name'))
                    pre_candidate_ids.add(candidate_id)
                    # Calculate percentage uniformly for all test files
                    total_score = sum(candidate['marks'].values())
                    pre_scores[candidate_id] = (total_score / 20) * 100 if candidate['marks'] and len(candidate['marks']) > 0 and 20 > 0 else 0
                
                for candidate in post_candidates:
                    # Use ticket_no or name for candidate identification
                    candidate_id = candidate.get('ticket_no', candidate.get('name'))
                    post_candidate_ids.add(candidate_id)
                    # Calculate percentage uniformly for all test files
                    total_score = sum(candidate['marks'].values())
                    post_scores[candidate_id] = (total_score / 20) * 100 if candidate['marks'] and len(candidate['marks']) > 0 and 20 > 0 else 0
                
                # Find common candidates (attended both tests)
                common_candidates = pre_candidate_ids.intersection(post_candidate_ids)
                
                # Find candidates who only attended pre-test
                pre_only_candidates = pre_candidate_ids.difference(post_candidate_ids)
                
                # Find candidates who only attended post-test
                post_only_candidates = post_candidate_ids.difference(pre_candidate_ids)
                
                # Calculate improvements for common candidates
                improvements = []
                total_improvement = 0
                
                # Additional statistics for deeper analysis
                pre_test_scores_list = []
                post_test_scores_list = []
                
                for candidate_id in common_candidates:
                    pre_score = pre_scores[candidate_id]
                    post_score = post_scores[candidate_id]
                    improvement = post_score - pre_score
                    
                    pre_test_scores_list.append(pre_score)
                    post_test_scores_list.append(post_score)
                    
                    improvements.append({
                        'candidate_id': candidate_id,
                        'pre_test_score': round(pre_score, 2),
                        'post_test_score': round(post_score, 2),
                        'improvement': round(improvement, 2)
                    })
                    
                    total_improvement += improvement                
                # Sort by improvement (descending)
                improvements.sort(key=lambda x: x['improvement'], reverse=True)
                
                # Calculate statistics for common candidates
                avg_improvement = total_improvement / len(common_candidates) if common_candidates else 0
                positive_improvements = [imp for imp in improvements if imp['improvement'] > 0]
                negative_improvements = [imp for imp in improvements if imp['improvement'] < 0]
                
                # Calculate additional statistics for deeper analysis
                pre_test_scores_list = [imp['pre_test_score'] for imp in improvements]
                post_test_scores_list = [imp['post_test_score'] for imp in improvements]
                pre_test_avg = statistics.mean(pre_test_scores_list) if pre_test_scores_list else 0
                post_test_avg = statistics.mean(post_test_scores_list) if post_test_scores_list else 0
                pre_test_std = statistics.stdev(pre_test_scores_list) if len(pre_test_scores_list) > 1 else 0
                post_test_std = statistics.stdev(post_test_scores_list) if len(post_test_scores_list) > 1 else 0
                
                # Calculate improvement distribution
                significant_improvement = len([imp for imp in improvements if imp['improvement'] >= 10])
                moderate_improvement = len([imp for imp in improvements if 5 <= imp['improvement'] < 10])
                slight_improvement = len([imp for imp in improvements if 0 < imp['improvement'] < 5])
                no_improvement = len([imp for imp in improvements if imp['improvement'] == 0])
                declined = len([imp for imp in improvements if imp['improvement'] < 0])
                
                # Top and bottom performers
                top_improvers = improvements[:5]  # Top 5
                bottom_improvers = improvements[-5:] if len(improvements) >= 5 else improvements[::-1][:5]  # Bottom 5
                
                # Overall training effectiveness
                training_effectiveness_index = (avg_improvement / 100) * 100  # Normalize to 0-100 scale
                
                # Calculate Learning Index components
                learning_gain = avg_improvement
                consistency_index = 100 - (abs(pre_test_std - post_test_std) / max(pre_test_std, post_test_std, 1)) * 100 if pre_test_std > 0 or post_test_std > 0 else 100
                improvement_rate = (len(positive_improvements) / len(common_candidates)) * 100 if common_candidates else 0
                
                # Comprehensive Learning Index (0-100 scale)
                learning_index = (learning_gain * 0.5) + (consistency_index * 0.3) + (improvement_rate * 0.2) if learning_gain >= 0 else (learning_gain * 0.7) + (consistency_index * 0.2) + (improvement_rate * 0.1)                
                # Prepare data for candidates who only attended pre-test
                pre_only_data = []
                for candidate_id in pre_only_candidates:
                    pre_only_data.append({
                        'candidate_id': candidate_id,
                        'pre_test_score': round(pre_scores[candidate_id], 2)
                    })
                
                # Prepare data for candidates who only attended post-test
                post_only_data = []
                for candidate_id in post_only_candidates:
                    post_only_data.append({
                        'candidate_id': candidate_id,
                        'post_test_score': round(post_scores[candidate_id], 2)
                    })
                
                result = {
                    "comparison": {
                        "pre_test": {
                            "test_id": pre_test_id,
                            "test_type": pre_test.test_type,
                            "training_name": pre_test.training_name
                        },
                        "post_test": {
                            "test_id": post_test_id,
                            "test_type": post_test.test_type,
                            "training_name": post_test.training_name
                        },
                        "statistics": {
                            "total_candidates_compared": len(common_candidates),
                            "average_improvement": round(avg_improvement, 2),
                            "positive_improvements": len(positive_improvements),
                            "negative_improvements": len(negative_improvements),
                            "no_change": len(common_candidates) - len(positive_improvements) - len(negative_improvements),
                            "training_effectiveness_index": round(training_effectiveness_index, 2),
                            "learning_index": round(learning_index, 2),
                            "pre_test_average": round(pre_test_avg, 2),
                            "post_test_average": round(post_test_avg, 2),
                            "pre_test_std_deviation": round(pre_test_std, 2),
                            "post_test_std_deviation": round(post_test_std, 2)
                        },
                        "improvement_distribution": {
                            "significant_improvement": significant_improvement,
                            "moderate_improvement": moderate_improvement,
                            "slight_improvement": slight_improvement,
                            "no_improvement": no_improvement,
                            "declined": declined
                        },
                        "attendance_analysis": {
                            "attended_both_tests": len(common_candidates),
                            "attended_pre_only": len(pre_only_candidates),
                            "attended_post_only": len(post_only_candidates)
                        },
                        "top_improvers": top_improvers,
                        "bottom_improvers": bottom_improvers,
                        "all_improvements": improvements,
                        "pre_test_only_candidates": pre_only_data,
                        "post_test_only_candidates": post_only_data
                    }
                }
                
                db_session.close()
                return jsonify(result), 200
            else:
                db_session.close()
                return jsonify({"error": "Both tests must be evaluated first or have valid Excel files"}), 400
        else:
            # Use existing evaluations
            # Create dictionaries for easier comparison
            pre_scores = {eval.candidate_id: eval.percentage for eval in pre_evaluations}
            post_scores = {eval.candidate_id: eval.percentage for eval in post_evaluations}
            
            # Collect all candidate IDs
            pre_candidate_ids = set(pre_scores.keys())
            post_candidate_ids = set(post_scores.keys())
            
            # Find common candidates (attended both tests)
            common_candidates = pre_candidate_ids.intersection(post_candidate_ids)
            
            # Find candidates who only attended pre-test
            pre_only_candidates = pre_candidate_ids.difference(post_candidate_ids)
            
            # Find candidates who only attended post-test
            post_only_candidates = post_candidate_ids.difference(pre_candidate_ids)
            
            # Calculate improvements for common candidates
            improvements = []
            total_improvement = 0
            
            for candidate_id in common_candidates:
                pre_score = pre_scores[candidate_id]
                post_score = post_scores[candidate_id]
                improvement = post_score - pre_score
                
                improvements.append({
                    'candidate_id': candidate_id,
                    'pre_test_score': pre_score,
                    'post_test_score': post_score,
                    'improvement': improvement
                })
                
                total_improvement += improvement
            
            # Sort by improvement (descending)
            improvements.sort(key=lambda x: x['improvement'], reverse=True)
            
            # Calculate additional statistics for deeper analysis
            pre_test_scores_list = [imp['pre_test_score'] for imp in improvements]
            post_test_scores_list = [imp['post_test_score'] for imp in improvements]
            pre_test_avg = statistics.mean(pre_test_scores_list) if pre_test_scores_list else 0
            post_test_avg = statistics.mean(post_test_scores_list) if post_test_scores_list else 0
            pre_test_std = statistics.stdev(pre_test_scores_list) if len(pre_test_scores_list) > 1 else 0
            post_test_std = statistics.stdev(post_test_scores_list) if len(post_test_scores_list) > 1 else 0
            
            # Calculate improvement distribution
            significant_improvement = len([imp for imp in improvements if imp['improvement'] >= 10])
            moderate_improvement = len([imp for imp in improvements if 5 <= imp['improvement'] < 10])
            slight_improvement = len([imp for imp in improvements if 0 < imp['improvement'] < 5])
            no_improvement = len([imp for imp in improvements if imp['improvement'] == 0])
            declined = len([imp for imp in improvements if imp['improvement'] < 0])
            
            # Calculate statistics for common candidates
            avg_improvement = total_improvement / len(common_candidates) if common_candidates and len(common_candidates) > 0 else 0
            positive_improvements = [imp for imp in improvements if imp['improvement'] > 0]
            negative_improvements = [imp for imp in improvements if imp['improvement'] < 0]
            
            # Top and bottom performers
            top_improvers = improvements[:5]  # Top 5
            bottom_improvers = improvements[-5:] if len(improvements) >= 5 else improvements[::-1][:5]  # Bottom 5
            
            # Overall training effectiveness
            training_effectiveness_index = (avg_improvement / 100) * 100  # Normalize to 0-100 scale
            
            # Calculate Learning Index components
            learning_gain = avg_improvement
            consistency_index = 100 - (abs(pre_test_std - post_test_std) / max(pre_test_std, post_test_std, 1)) * 100 if pre_test_std > 0 or post_test_std > 0 else 100
            improvement_rate = (len(positive_improvements) / len(common_candidates)) * 100 if common_candidates and len(common_candidates) > 0 else 0
            
            # Comprehensive Learning Index (0-100 scale)
            learning_index = (learning_gain * 0.5) + (consistency_index * 0.3) + (improvement_rate * 0.2) if learning_gain >= 0 else (learning_gain * 0.7) + (consistency_index * 0.2) + (improvement_rate * 0.1)
            
            # Prepare data for candidates who only attended pre-test
            pre_only_data = []
            for candidate_id in pre_only_candidates:
                pre_only_data.append({
                    'candidate_id': candidate_id,
                    'pre_test_score': pre_scores[candidate_id]
                })
            
            # Prepare data for candidates who only attended post-test
            post_only_data = []
            for candidate_id in post_only_candidates:
                post_only_data.append({
                    'candidate_id': candidate_id,
                    'post_test_score': post_scores[candidate_id]
                })
            
            result = {
                "comparison": {
                    "pre_test": {
                        "test_id": pre_test_id,
                        "test_type": pre_test.test_type,
                        "training_name": pre_test.training_name
                    },
                    "post_test": {
                        "test_id": post_test_id,
                        "test_type": post_test.test_type,
                        "training_name": post_test.training_name
                    },
                    "statistics": {
                        "total_candidates_compared": len(common_candidates),
                        "average_improvement": round(avg_improvement, 2),
                        "positive_improvements": len(positive_improvements),
                        "negative_improvements": len(negative_improvements),
                        "no_change": len(common_candidates) - len(positive_improvements) - len(negative_improvements),
                        "training_effectiveness_index": round(training_effectiveness_index, 2),
                        "learning_index": round(learning_index, 2),
                        "pre_test_average": round(pre_test_avg, 2),
                        "post_test_average": round(post_test_avg, 2),
                        "pre_test_std_deviation": round(pre_test_std, 2),
                        "post_test_std_deviation": round(post_test_std, 2)
                    },
                    "improvement_distribution": {
                        "significant_improvement": significant_improvement,
                        "moderate_improvement": moderate_improvement,
                        "slight_improvement": slight_improvement,
                        "no_improvement": no_improvement,
                        "declined": declined
                    },
                    "attendance_analysis": {
                        "attended_both_tests": len(common_candidates),
                        "attended_pre_only": len(pre_only_candidates),
                        "attended_post_only": len(post_only_candidates)
                    },
                    "top_improvers": top_improvers,
                    "bottom_improvers": bottom_improvers,
                    "all_improvements": improvements,
                    "pre_test_only_candidates": pre_only_data,
                    "post_test_only_candidates": post_only_data
                }
            }
            
            db_session.close()
            return jsonify(result), 200
            
    except Exception as e:
        return jsonify({"error": f"Comparison failed: {str(e)}"}), 500

@app.route('/api/analytics/historical/training/<training_name>', methods=['GET'])
def get_historical_analytics_by_training(training_name):
    """Get historical analytics for a specific training name"""
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        # Get all tests with the specified training name
        tests = db_session.query(Test).filter(Test.training_name == training_name).all()
        
        if not tests:
            db_session.close()
            return jsonify({"error": "No tests found for this training name"}), 404
        
        # Collect all evaluations for these tests
        all_evaluations = []
        test_details = []
        
        for test in tests:
            evaluations = db_session.query(Evaluation).filter(Evaluation.test_id == test.test_id).all()
            all_evaluations.extend(evaluations)
            
            test_details.append({
                "test_id": test.test_id,
                "test_type": test.test_type,
                "date_uploaded": test.date_uploaded.isoformat() if test.date_uploaded else None,
                "total_candidates": len(evaluations)
            })
        
        # Calculate statistics
        scores = [eval.total_score for eval in all_evaluations]
        percentages = [eval.percentage for eval in all_evaluations]
        
        if not scores:
            db_session.close()
            return jsonify({"error": "No evaluation data found for this training"}), 404
        
        # Calculate overall statistics
        overall_stats = {
            "average_score": statistics.mean(scores) if scores else 0,
            "median_score": statistics.median(scores) if scores else 0,
            "highest_score": max(scores) if scores else 0,
            "lowest_score": min(scores) if scores else 0,
            "pass_rate": (len([s for s in scores if s >= 40]) / len(scores)) * 100 if scores and len(scores) > 0 else 0,
            "total_candidates": len(scores),
            "total_tests": len(tests)
        }
        
        # Performance distribution
        excellent = len([p for p in percentages if p >= 90])
        good = len([p for p in percentages if 70 <= p < 90])
        average = len([p for p in percentages if 50 <= p < 70])
        poor = len([p for p in percentages if p < 50])
        
        performance_distribution = {
            "excellent": excellent,
            "good": good,
            "average": average,
            "poor": poor
        }
        
        result = {
            "training_name": training_name,
            "overall_stats": overall_stats,
            "performance_distribution": performance_distribution,
            "test_history": test_details
        }
        
        db_session.close()
        return jsonify(result)
        
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/historical/date-range', methods=['GET'])
def get_historical_analytics_by_date_range():
    """Get historical analytics for a specific date range"""
    # Get date range parameters
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    if not start_date_str or not end_date_str:
        return jsonify({"error": "Both start_date and end_date parameters are required"}), 400
    
    try:
        from datetime import datetime
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format. Use ISO format (YYYY-MM-DD)"}), 400
    
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        # Get all tests within the date range
        tests = db_session.query(Test).filter(
            Test.date_uploaded >= start_date,
            Test.date_uploaded <= end_date
        ).all()
        
        if not tests:
            db_session.close()
            return jsonify({"error": "No tests found for this date range"}), 404
        
        # Collect all evaluations for these tests
        all_evaluations = []
        test_details = []
        
        for test in tests:
            evaluations = db_session.query(Evaluation).filter(Evaluation.test_id == test.test_id).all()
            all_evaluations.extend(evaluations)
            
            test_details.append({
                "test_id": test.test_id,
                "test_type": test.test_type,
                "training_name": test.training_name,
                "date_uploaded": test.date_uploaded.isoformat() if test.date_uploaded else None,
                "total_candidates": len(evaluations)
            })
        
        # Calculate statistics
        scores = [eval.total_score for eval in all_evaluations]
        percentages = [eval.percentage for eval in all_evaluations]
        
        if not scores:
            db_session.close()
            return jsonify({"error": "No evaluation data found for this date range"}), 404
        
        # Calculate overall statistics
        overall_stats = {
            "average_score": statistics.mean(scores) if scores else 0,
            "median_score": statistics.median(scores) if scores else 0,
            "highest_score": max(scores) if scores else 0,
            "lowest_score": min(scores) if scores else 0,
            "pass_rate": (len([s for s in scores if s >= 40]) / len(scores)) * 100 if scores and len(scores) > 0 else 0,
            "total_candidates": len(scores),
            "total_tests": len(tests)
        }
        
        # Performance distribution
        excellent = len([p for p in percentages if p >= 90])
        good = len([p for p in percentages if 70 <= p < 90])
        average = len([p for p in percentages if 50 <= p < 70])
        poor = len([p for p in percentages if p < 50])
        
        performance_distribution = {
            "excellent": excellent,
            "good": good,
            "average": average,
            "poor": poor
        }
        
        # Group by training name
        training_summary = {}
        for test in tests:
            name = test.training_name
            if name not in training_summary:
                training_summary[name] = {
                    "training_name": name,
                    "test_count": 0,
                    "total_candidates": 0
                }
            training_summary[name]["test_count"] += 1
            
            # Add candidates for this test
            evaluations = db_session.query(Evaluation).filter(Evaluation.test_id == test.test_id).all()
            training_summary[name]["total_candidates"] += len(evaluations)
        
        result = {
            "date_range": {
                "start_date": start_date_str,
                "end_date": end_date_str
            },
            "overall_stats": overall_stats,
            "performance_distribution": performance_distribution,
            "test_history": test_details,
            "training_summary": list(training_summary.values())
        }
        
        db_session.close()
        return jsonify(result)
        
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/historical/faculty/<name>', methods=['GET'])
def get_historical_analytics_by_faculty(name):
    """Get historical analytics for a specific faculty name"""
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        # Get all tests with this faculty name (exact match)
        tests = db_session.query(Test).filter(
            Test.faculty_name == name
        ).all()
        
        if not tests:
            db_session.close()
            return jsonify({"error": f"No tests found for faculty matching '{name}'"}), 404
        
        # Collect all evaluations for these tests
        all_evaluations = []
        test_details = []
        
        for test in tests:
            evaluations = db_session.query(Evaluation).filter(Evaluation.test_id == test.test_id).all()
            all_evaluations.extend(evaluations)
            
            test_details.append({
                "test_id": test.test_id,
                "test_type": test.test_type,
                "training_name": test.training_name,
                "date_uploaded": test.date_uploaded.isoformat() if test.date_uploaded else None,
                "total_candidates": len(evaluations)
            })
        
        # Calculate faculty-specific statistics
        faculty_scores = [eval.total_score for eval in all_evaluations]
        faculty_percentages = [eval.percentage for eval in all_evaluations]
        
        overall_stats = {
            "average_score": statistics.mean(faculty_scores) if faculty_scores else 0,
            "median_score": statistics.median(faculty_scores) if faculty_scores else 0,
            "highest_score": max(faculty_scores) if faculty_scores else 0,
            "lowest_score": min(faculty_scores) if faculty_scores else 0,
            "pass_rate": (len([s for s in faculty_scores if s >= 40]) / len(faculty_scores)) * 100 if faculty_scores and len(faculty_scores) > 0 else 0,
            "total_tests": len(tests),
            "total_candidates": len(all_evaluations)
        }
        
        # Performance distribution
        excellent = len([p for p in faculty_percentages if p >= 90])
        good = len([p for p in faculty_percentages if 70 <= p < 90])
        average = len([p for p in faculty_percentages if 50 <= p < 70])
        poor = len([p for p in faculty_percentages if p < 50])
        
        performance_distribution = {
            "excellent": excellent,
            "good": good,
            "average": average,
            "poor": poor
        }
        
        result = {
            "name": name,
            "overall_stats": overall_stats,
            "performance_distribution": performance_distribution,
            "test_history": test_details
        }
        
        db_session.close()
        return jsonify(result)
        
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/historical/filtered', methods=['GET'])
def get_filtered_historical_analytics():
    """Get historical analytics with multiple filters (training_name, date_range, faculty_name, ticket_no)"""
    # Get filter parameters
    training_name = request.args.get('training_name')
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    faculty_name = request.args.get('faculty_name')
    ticket_no = request.args.get('ticket_no')
    
    # Create a new database session for this request
    db_session = get_db_session()
    
    try:
        # Build query based on filters
        # We need to distinguish between batch ticket filtering and candidate ID filtering
        # For now, we'll assume ticket_no refers to batch_ticket_no when used alone
        query = db_session.query(Test)
        
        # Apply training name filter
        if training_name:
            query = query.filter(Test.training_name == training_name)
        
        # Apply date range filter (using date_conducted instead of date_uploaded)
        if start_date_str and end_date_str:
            try:
                from datetime import datetime
                start_date = datetime.fromisoformat(start_date_str)
                end_date = datetime.fromisoformat(end_date_str)
                query = query.filter(
                    Test.date_conducted >= start_date,
                    Test.date_conducted <= end_date
                )
            except ValueError:
                db_session.close()
                return jsonify({"error": "Invalid date format. Use ISO format (YYYY-MM-DD)"}), 400
        
        # Apply faculty name filter
        if faculty_name:
            # Filter tests that have matching faculty names (case-sensitive exact match)
            query = query.filter(Test.faculty_name == faculty_name)
        
        # Apply ticket_no filter - assume it's for batch filtering
        if ticket_no:
            query = query.filter(
                (Test.batch_ticket_no == ticket_no) | (Test.batch_ticket_no.like(f"%{ticket_no}%"))
            )
        
        # Execute query
        tests = query.all()
        
        if not tests:
            # Return empty result instead of 404 when no tests match filters
            result = {
                "filters_applied": {
                    "training_name": training_name,
                    "date_range": {
                        "start_date": start_date_str,
                        "end_date": end_date_str
                    } if start_date_str and end_date_str else None,
                    "faculty_name": faculty_name,
                    "ticket_no": ticket_no
                },
                "overall_stats": {
                    "average_score": 0,
                    "median_score": 0,
                    "highest_score": 0,
                    "lowest_score": 0,
                    "pass_rate": 0,
                    "total_candidates": 0,
                    "total_tests": 0
                },
                "performance_distribution": {
                    "excellent": 0,
                    "good": 0,
                    "average": 0,
                    "poor": 0
                },
                "test_history": [],
                "training_summary": []
            }
            db_session.close()
            return jsonify(result)
        
        # Collect all evaluations for these tests
        all_evaluations = []
        test_details = []
        
        for test in tests:
            # When ticket_no is provided, we assume it's for batch filtering
            # So we get all evaluations for tests with matching batch_ticket_no
            total_evaluations = db_session.query(Evaluation).filter(Evaluation.test_id == test.test_id).all()
            all_evaluations.extend(total_evaluations)
            
            # Add test details
            test_details.append({
                "test_id": test.test_id,
                "test_type": test.test_type,
                "training_name": test.training_name,
                "faculty_name": test.faculty_name,
                "date_uploaded": test.date_conducted.isoformat() if test.date_conducted else None,
                "total_candidates": len(total_evaluations),
                "status": test.status
            })
        
        # Calculate statistics
        scores = [eval.total_score for eval in all_evaluations]
        percentages = [eval.percentage for eval in all_evaluations]
        
        if not scores:
            # Return result with empty stats when no evaluation data found
            result = {
                "filters_applied": {
                    "training_name": training_name,
                    "date_range": {
                        "start_date": start_date_str,
                        "end_date": end_date_str
                    } if start_date_str and end_date_str else None,
                    "faculty_name": faculty_name,
                    "ticket_no": ticket_no
                },
                "overall_stats": {
                    "average_score": 0,
                    "median_score": 0,
                    "highest_score": 0,
                    "lowest_score": 0,
                    "pass_rate": 0,
                    "total_candidates": 0,
                    "total_tests": len(test_details)
                },
                "performance_distribution": {
                    "excellent": 0,
                    "good": 0,
                    "average": 0,
                    "poor": 0
                },
                "test_history": test_details,
                "training_summary": []
            }
            db_session.close()
            return jsonify(result)
        
        # Calculate overall statistics
        overall_stats = {
            "average_score": statistics.mean(scores) if scores else 0,
            "median_score": statistics.median(scores) if scores else 0,
            "highest_score": max(scores) if scores else 0,
            "lowest_score": min(scores) if scores else 0,
            "pass_rate": (len([p for p in percentages if p >= 40]) / len(percentages)) * 100 if percentages and len(percentages) > 0 else 0,
            "total_candidates": len(scores),
            "total_tests": len(test_details)  # Use actual displayed tests, not all filtered tests
        }
        
        # Performance distribution
        excellent = len([p for p in percentages if p >= 90])
        good = len([p for p in percentages if 70 <= p < 90])
        average = len([p for p in percentages if 50 <= p < 70])
        poor = len([p for p in percentages if p < 50])
        
        performance_distribution = {
            "excellent": excellent,
            "good": good,
            "average": average,
            "poor": poor
        }
        
        # Group by training name for summary
        training_summary = {}
        for test in tests:
            name = test.training_name
            if name not in training_summary:
                training_summary[name] = {
                    "training_name": name,
                    "test_count": 0,
                    "total_candidates": 0
                }
            training_summary[name]["test_count"] += 1
            
            # Add candidates for this test
            evaluations = db_session.query(Evaluation).filter(Evaluation.test_id == test.test_id).all()
            training_summary[name]["total_candidates"] += len(evaluations)
        
        result = {
            "filters_applied": {
                "training_name": training_name,
                "date_range": {
                    "start_date": start_date_str,
                    "end_date": end_date_str
                } if start_date_str and end_date_str else None,
                "faculty_name": faculty_name,
                "ticket_no": ticket_no
            },
            "overall_stats": overall_stats,
            "performance_distribution": performance_distribution,
            "test_history": test_details,
            "training_summary": list(training_summary.values())
        }
        
        db_session.close()
        return jsonify(result)
        
    except Exception as e:
        db_session.close()
        return jsonify({"error": str(e)}), 500

# Error handlers

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

# Manual evaluation endpoint removed - all evaluations are now automatic
# This endpoint has been removed to ensure consistent automated test evaluation
# All tests are evaluated automatically during the upload process

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
