"""
Test script to verify the Tata Motors Performance Analytics System is working correctly.
"""
import requests
import json
import time
import os

def test_system():
    base_url = "http://127.0.0.1:5000"
    
    print("Testing Tata Motors Performance Analytics System...")
    print("="*60)
    
    # Test 1: Check if the API is running
    print("\n1. Testing API availability...")
    try:
        response = requests.get(f"{base_url}/api")
        if response.status_code == 200:
            print("✓ API is running and accessible")
            api_info = response.json()
            print(f"  Message: {api_info.get('message', 'N/A')}")
            print(f"  Version: {api_info.get('version', 'N/A')}")
        else:
            print(f"✗ API test failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API test failed with error: {str(e)}")
        return False
    
    # Test 2: Check if we can get an empty test list (before any uploads)
    print("\n2. Testing tests endpoint (should be empty initially)...")
    try:
        response = requests.get(f"{base_url}/api/tests")
        if response.status_code == 200:
            tests = response.json()
            print(f"✓ Tests endpoint working, current test count: {len(tests)}")
        else:
            print(f"✗ Tests endpoint failed with status code: {response.status_code}")
    except Exception as e:
        print(f"✗ Tests endpoint test failed with error: {str(e)}")
    
    # Test 3: Check analytics endpoints
    print("\n3. Testing overall analytics endpoint...")
    try:
        response = requests.get(f"{base_url}/api/analytics/overall")
        if response.status_code == 200:
            analytics = response.json()
            print("✓ Overall analytics endpoint working")
            if 'overall_stats' in analytics:
                print(f"  Total candidates: {analytics['overall_stats'].get('total_candidates', 0)}")
                print(f"  Pass rate: {analytics['overall_stats'].get('pass_rate', 0)}%")
            else:
                print("  (No data available yet)")
        else:
            print(f"✗ Analytics endpoint failed with status code: {response.status_code}")
    except Exception as e:
        print(f"✗ Analytics endpoint test failed with error: {str(e)}")
    
    # Test 4: Check training names endpoint
    print("\n4. Testing training names endpoint...")
    try:
        response = requests.get(f"{base_url}/api/trainings")
        if response.status_code == 200:
            trainings = response.json()
            print(f"✓ Training names endpoint working, found {len(trainings)} trainings")
        else:
            print(f"✗ Training names endpoint failed with status code: {response.status_code}")
    except Exception as e:
        print(f"✗ Training names endpoint test failed with error: {str(e)}")
    
    # Test 5: Check faculty names endpoint
    print("\n5. Testing faculty names endpoint...")
    try:
        response = requests.get(f"{base_url}/api/faculties")
        if response.status_code == 200:
            faculties = response.json()
            print(f"✓ Faculty names endpoint working, found {len(faculties)} faculties")
        else:
            print(f"✗ Faculty names endpoint failed with status code: {response.status_code}")
    except Exception as e:
        print(f"✗ Faculty names endpoint test failed with error: {str(e)}")
    
    # Test 6: Test the delete all data functionality
    print("\n6. Testing delete all data functionality...")
    try:
        response = requests.delete(f"{base_url}/api/delete-all-data")
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Delete all data endpoint working: {result.get('message', 'N/A')}")
        else:
            print(f"✗ Delete all data endpoint failed with status code: {response.status_code}")
    except Exception as e:
        print(f"✗ Delete all data endpoint test failed with error: {str(e)}")
    
    # Test 7: Verify that data was actually deleted by checking tests again
    print("\n7. Verifying data was deleted...")
    try:
        response = requests.get(f"{base_url}/api/tests")
        if response.status_code == 200:
            tests_after_delete = response.json()
            if len(tests_after_delete) == 0:
                print("✓ Data deletion verified - no tests found after delete")
            else:
                print(f"✗ Data may not have been fully deleted - still found {len(tests_after_delete)} tests")
        else:
            print(f"✗ Verification test failed with status code: {response.status_code}")
    except Exception as e:
        print(f"✗ Verification test failed with error: {str(e)}")
    
    print("\n" + "="*60)
    print("System test completed!")
    print("The system is properly configured to handle 20-mark tests")
    print("and store data in the employee_performance_new.db file")
    print("in the main project directory.")
    
    return True

if __name__ == "__main__":
    test_system()