"""
Test script specifically for verifying the delete all Excel data functionality.
"""
import requests
import json
import time
import os

def test_delete_functionality():
    base_url = "http://127.0.0.1:5000"
    
    print("Testing Delete All Excel Data Functionality...")
    print("="*60)
    
    # Step 1: Check initial state
    print("\n1. Checking initial state...")
    try:
        response = requests.get(f"{base_url}/api/tests")
        if response.status_code == 200:
            initial_tests = response.json()
            print(f"✓ Found {len(initial_tests)} tests initially")
        else:
            print(f"✗ Failed to get initial tests, status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error getting initial tests: {str(e)}")
        return False
    
    # Step 2: Check initial analytics
    print("\n2. Checking initial analytics...")
    try:
        response = requests.get(f"{base_url}/api/analytics/overall")
        if response.status_code == 200:
            initial_analytics = response.json()
            if 'overall_stats' in initial_analytics:
                initial_candidates = initial_analytics['overall_stats'].get('total_candidates', 0)
                print(f"✓ Found {initial_candidates} candidates initially")
            else:
                print("✓ No analytics data found initially")
                initial_candidates = 0
        else:
            print(f"✗ Failed to get initial analytics, status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error getting initial analytics: {str(e)}")
        return False
    
    # Step 3: Execute delete all data
    print("\n3. Executing delete all data...")
    try:
        response = requests.delete(f"{base_url}/api/delete-all-data")
        if response.status_code == 200:
            delete_result = response.json()
            print(f"✓ Delete operation successful: {delete_result.get('message', 'N/A')}")
        else:
            print(f"✗ Delete operation failed, status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error executing delete operation: {str(e)}")
        return False
    
    # Step 4: Wait briefly to ensure deletion is processed
    time.sleep(1)
    
    # Step 5: Verify tests are deleted
    print("\n4. Verifying tests are deleted...")
    try:
        response = requests.get(f"{base_url}/api/tests")
        if response.status_code == 200:
            tests_after_delete = response.json()
            if len(tests_after_delete) == 0:
                print("✓ All tests successfully deleted")
            else:
                print(f"✗ {len(tests_after_delete)} tests still remain after delete")
                return False
        else:
            print(f"✗ Failed to verify tests after delete, status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error verifying tests after delete: {str(e)}")
        return False
    
    # Step 6: Verify analytics are cleared
    print("\n5. Verifying analytics are cleared...")
    try:
        response = requests.get(f"{base_url}/api/analytics/overall")
        if response.status_code == 200:
            analytics_after_delete = response.json()
            if 'overall_stats' in analytics_after_delete:
                candidates_after_delete = analytics_after_delete['overall_stats'].get('total_candidates', 0)
                if candidates_after_delete == 0:
                    print("✓ All analytics data successfully cleared")
                else:
                    print(f"✗ {candidates_after_delete} candidates still found after delete")
                    return False
            else:
                print("✓ Analytics data structure is empty after delete")
        else:
            print(f"✗ Failed to verify analytics after delete, status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error verifying analytics after delete: {str(e)}")
        return False
    
    # Step 7: Verify trainings are cleared
    print("\n6. Verifying trainings are cleared...")
    try:
        response = requests.get(f"{base_url}/api/trainings")
        if response.status_code == 200:
            trainings_after_delete = response.json()
            if len(trainings_after_delete) == 0:
                print("✓ All training data successfully cleared")
            else:
                print(f"✗ {len(trainings_after_delete)} trainings still found after delete")
                return False
        else:
            print(f"✗ Failed to verify trainings after delete, status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error verifying trainings after delete: {str(e)}")
        return False
    
    # Step 8: Verify faculties are cleared
    print("\n7. Verifying faculties are cleared...")
    try:
        response = requests.get(f"{base_url}/api/faculties")
        if response.status_code == 200:
            faculties_after_delete = response.json()
            if len(faculties_after_delete) == 0:
                print("✓ All faculty data successfully cleared")
            else:
                print(f"✗ {len(faculties_after_delete)} faculties still found after delete")
                return False
        else:
            print(f"✗ Failed to verify faculties after delete, status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error verifying faculties after delete: {str(e)}")
        return False
    
    print("\n" + "="*60)
    print("Delete All Excel Data functionality test completed successfully!")
    print("✓ All Excel data, tests, analytics, trainings, and faculties have been cleared")
    print("✓ The delete functionality is working correctly")
    
    return True

if __name__ == "__main__":
    test_delete_functionality()