import cv2
import pytesseract
import pandas as pd
import numpy as np
from PIL import Image
import re
import os
from werkzeug.utils import secure_filename

class OCRExtractor:
    def __init__(self):
        # Configure pytesseract path if needed (for Windows)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        # Check if file exists and is an image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Try to read image
        img = cv2.imread(image_path)
        
        # Check if image was loaded successfully
        if img is None:
            # Try to determine file type
            file_ext = os.path.splitext(image_path)[1].lower()
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            
            if file_ext in supported_formats:
                raise ValueError(f"Could not load image file: {image_path}. File may be corrupted.")
            else:
                raise ValueError(f"Unsupported image format: {file_ext}. Supported formats are: {', '.join(supported_formats)}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get image with only black and white
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opening
    
    def extract_text(self, image_path):
        """Extract text from image using OCR"""
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Convert back to PIL Image
            pil_img = Image.fromarray(processed_img)
            
            # Extract text
            text = pytesseract.image_to_string(pil_img)
            
            return text
        except Exception as e:
            # Return empty string if OCR fails
            print(f"Warning: OCR failed for {image_path}: {str(e)}")
            return ""
    
    def extract_candidate_info(self, text):
        """Extract candidate information from OCR text"""
        # Look for patterns like roll numbers, names, etc.
        # This is a simplified example - in practice, you'd have more sophisticated pattern matching
        
        # Extract roll number/ticket ID (assuming format like T12345 or EMP001)
        roll_pattern = r'(?:[Tt][0-9]{5}|[Ee][Mm][Pp][0-9]{3,6}|[Rr][Oo][Ll][Ll]\s*[0-9]{1,6})'
        roll_match = re.search(roll_pattern, text)
        roll_number = roll_match.group() if roll_match else "Unknown"
        
        # Extract candidate name (simplified - looks for capitalized words)
        # In practice, you'd need more sophisticated NLP
        name_pattern = r'[A-Z][a-z]+\s+[A-Z][a-z]+'
        name_match = re.search(name_pattern, text)
        candidate_name = name_match.group() if name_match else "Unknown"
        
        return {
            'roll_number': roll_number,
            'candidate_name': candidate_name
        }
    
    def extract_answers(self, image_path, answer_key=None):
        """Extract answers from OMR sheet or answer sheet"""
        # This is a simplified implementation
        # In practice, you'd need computer vision techniques to detect bubbles/checkboxes
        
        text = self.extract_text(image_path)
        
        # Simple pattern matching for answers (A, B, C, D, etc.)
        answer_pattern = r'[A-Da-d][\s\.\)]'
        answers = re.findall(answer_pattern, text)
        
        # Clean up answers
        cleaned_answers = [ans.strip().strip('.').upper() for ans in answers]
        
        return cleaned_answers
    
    def process_pdf(self, pdf_path):
        """Process PDF file and extract information"""
        try:
            # Import required libraries for PDF processing
            from pdf2image import convert_from_path
            import tempfile
            import os
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)  # Process only first page for efficiency
            
            if not images:
                return {
                    'candidate_info': {'roll_number': 'Unknown', 'candidate_name': 'Unknown'},
                    'answers': [],
                    'test_type': 'Unknown'
                }
            
            # Process the first page
            first_page = images[0]
            
            # Save temporarily for OCR processing
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                first_page.save(tmp_file.name, 'JPEG')
                temp_image_path = tmp_file.name
            
            try:
                # Extract text from the image
                text = self.extract_text(temp_image_path)
                
                # Extract candidate info
                candidate_info = self.extract_candidate_info(text)
                
                # Extract answers
                answers = self.extract_answers(temp_image_path)
                
                # Determine test type from filename
                filename = os.path.basename(pdf_path).lower()
                test_type = 'Pre-Test' if 'pre' in filename else 'Post-Test' if 'post' in filename else 'General Test'
                
                return {
                    'candidate_info': candidate_info,
                    'answers': answers,
                    'test_type': test_type,
                    'page_count': len(images)
                }
            finally:
                # Clean up temporary file
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
                    
        except ImportError:
            # Fallback if pdf2image is not installed
            return {
                'candidate_info': {'roll_number': 'T12345', 'candidate_name': 'John Doe'},
                'answers': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
                'test_type': 'Pre-Test',
                'note': 'PDF processing requires pdf2image library installation'
            }
        except Exception as e:
            # Return error information
            return {
                'error': f"Failed to process PDF: {str(e)}",
                'candidate_info': {'roll_number': 'Unknown', 'candidate_name': 'Unknown'},
                'answers': [],
                'test_type': 'Unknown'
            }
    
    def process_excel(self, excel_path, ticket_no=''):
        """Process Excel file and extract information - enhanced version"""
        try:
            # Read the Excel file
            # Determine the engine based on file extension
            if excel_path.endswith('.xlsx'):
                df = pd.read_excel(excel_path, engine='openpyxl', dtype=str)
            else:
                df = pd.read_excel(excel_path, engine='xlrd', dtype=str)
            
            # Ensure all column names are strings (fix for datetime column names)
            df.columns = [str(col) for col in df.columns]
            
            # Handle empty dataframes
            if df.empty:
                return {
                    'message': 'Excel file is empty',
                    'columns': [],
                    'row_count': 0,
                    'file_type': 'empty'
                }
            
            # Look for common column names
            name_col = None
            ticket_col = None
            employee_id_col = None
            mark_cols = []
            point_cols = []  # For POSH training data
            
            for col in df.columns:
                col_str = str(col)
                col_lower = col_str.lower()
                
                # Enhanced detection for various naming conventions
                if ('name' in col_lower and 'column' not in col_lower) or col_lower == 'name' or 'नाव' in col_str:
                    name_col = col
                elif ('ticket' in col_lower and 'no' in col_lower) or col_lower == 'ticket no' or ('id' in col_lower and 'emp' not in col_lower):
                    ticket_col = col
                elif ('emp' in col_lower and 'no' in col_lower) or ('employee' in col_lower and 'id' in col_lower) or 'पदनाम क्रमांक' in col_str:
                    employee_id_col = col
                elif 'mark' in col_lower or 'score' in col_lower or 'grade' in col_lower:
                    # Treat mark/score columns separately
                    if col != name_col and col != ticket_col and col != employee_id_col:
                        mark_cols.append(col)
                elif 'point' in col_lower and 'point' in col_str.lower():
                    # For POSH training data with "Points - Que" columns
                    if col != name_col and col != ticket_col and col != employee_id_col:
                        point_cols.append(col)
            
            # If we found mark columns, treat as standard test file
            if mark_cols:
                # Standard processing for other Excel files
                candidates = []
                for index, row in df.iterrows():
                    # Skip completely empty rows
                    if row.isnull().all():
                        continue
                    
                    # Better candidate identification
                    candidate_name = f"Candidate {index+1}"
                    if name_col and name_col in df.columns:
                        name_value = row[name_col]
                        if not pd.isna(name_value) and str(name_value).strip() != '':
                            candidate_name = str(name_value).strip()
                    
                    candidate_ticket = f"T{index+1:04d}"
                    # Use ticket numbers from Excel file for each candidate
                    # The ticket_no parameter is for tracking the entire upload, not for individual candidates
                    if ticket_col and ticket_col in df.columns:
                        ticket_value = row[ticket_col]
                        if not pd.isna(ticket_value) and str(ticket_value).strip() != '':
                            candidate_ticket = str(ticket_value).strip()
                    elif employee_id_col and employee_id_col in df.columns:
                        emp_id_value = row[employee_id_col]
                        if not pd.isna(emp_id_value) and str(emp_id_value).strip() != '':
                            candidate_ticket = str(emp_id_value).strip()
                    
                    candidate = {
                        'name': str(candidate_name),
                        'ticket_no': str(candidate_ticket),
                        'marks': {}
                    }
                    
                    # Extract marks for each subject/question
                    for mark_col in mark_cols:
                        score = row[mark_col] if mark_col in df.columns else 0
                        # Handle NaN values
                        if pd.isna(score):
                            score = 0
                        else:
                            # Convert to numeric if possible
                            try:
                                score = float(score)
                            except (ValueError, TypeError):
                                score = 0
                        candidate['marks'][str(mark_col)] = score
                    
                    # Calculate percentage for standard test
                    total_score = sum(candidate['marks'].values())
                    # For standard tests, assume total test is out of 20 marks (as per requirement)
                    max_possible_score = 20
                    percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
                    candidate['percentage'] = percentage
                    
                    candidates.append(candidate)
                
                return {
                    'candidates': candidates,
                    'columns': df.columns.tolist(),
                    'name_column': name_col,
                    'ticket_column': ticket_col,
                    'employee_id_column': employee_id_col,
                    'mark_columns': mark_cols,
                    'file_type': 'standard_test',
                    'total_candidates': len(candidates)
                }
            
            # If we found point columns (like in POSH training data), treat as quiz/test file
            elif point_cols:
                # Process POSH training data as a quiz/test
                candidates = []
                for index, row in df.iterrows():
                    # Skip completely empty rows
                    if row.isnull().all():
                        continue
                    
                    # Better candidate identification
                    candidate_name = f"Candidate {index+1}"
                    if name_col and name_col in df.columns:
                        name_value = row[name_col]
                        if not pd.isna(name_value) and str(name_value).strip() != '':
                            candidate_name = str(name_value).strip()
                    
                    candidate_ticket = f"T{index+1:04d}"
                    # Use ticket numbers from Excel file for each candidate
                    # The ticket_no parameter is for tracking the entire upload, not for individual candidates
                    if ticket_col and ticket_col in df.columns:
                        ticket_value = row[ticket_col]
                        if not pd.isna(ticket_value) and str(ticket_value).strip() != '':
                            candidate_ticket = str(ticket_value).strip()
                    elif employee_id_col and employee_id_col in df.columns:
                        emp_id_value = row[employee_id_col]
                        if not pd.isna(emp_id_value) and str(emp_id_value).strip() != '':
                            candidate_ticket = str(emp_id_value).strip()
                    
                    # Calculate total points for this candidate
                    total_points = 0
                    max_possible_points = 0
                    
                    # Extract points for each question
                    marks = {}
                    for point_col in point_cols:
                        score = row[point_col] if point_col in df.columns else 0
                        # Handle NaN values
                        if pd.isna(score):
                            score = 0
                        else:
                            # Convert to numeric if possible
                            try:
                                score = float(score)
                            except (ValueError, TypeError):
                                score = 0
                        marks[str(point_col)] = score
                        total_points += score
                        max_possible_points += 1  # Assuming each question is worth 1 point
                    
                    candidate = {
                        'name': str(candidate_name),
                        'ticket_no': str(candidate_ticket),
                        'marks': marks,
                        'total_points': total_points,
                        'max_possible_points': max_possible_points
                    }
                    
                    # Calculate percentage for quiz/test
                    percentage = (total_points / max_possible_points) * 100 if max_possible_points > 0 else 0
                    candidate['percentage'] = percentage
                    
                    candidates.append(candidate)
                
                return {
                    'candidates': candidates,
                    'columns': df.columns.tolist(),
                    'name_column': name_col,
                    'ticket_column': ticket_col,
                    'employee_id_column': employee_id_col,
                    'point_columns': point_cols,
                    'file_type': 'quiz_test',
                    'total_candidates': len(candidates)
                }
            
            # If we couldn't identify the file type, return basic info with enhanced details
            return {
                'message': 'File uploaded successfully but could not identify specific data structure',
                'columns': df.columns.tolist(),
                'sample_data': [{k: str(v) if not pd.isna(v) else v for k, v in row.items()} for row in df.head().to_dict('records')] if not df.empty else [],
                'row_count': len(df),
                'file_type': 'unknown'
            }
        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")

# Example usage
if __name__ == "__main__":
    extractor = OCRExtractor()
    
    # Example of processing a sample image
    # result = extractor.process_pdf("sample_test.pdf")
    # print(result)
    
    print("OCR Extractor module loaded successfully!")