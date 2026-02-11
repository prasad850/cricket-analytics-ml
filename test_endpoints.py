import requests
import json

def test_matches_endpoints():
    """Test both the HTML and API endpoints for matches"""
    print("🧪 Testing Matches Endpoints")
    print("=" * 40)
    
    try:
        # Test API endpoint
        print("Testing /api/matches endpoint...")
        response = requests.get("http://127.0.0.1:5000/api/matches", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API endpoint working!")
            
            # Count matches
            live_count = len(data.get('live', []))
            upcoming_count = len(data.get('upcoming', []))
            completed_count = len(data.get('completed', []))
            
            print(f"📊 Match counts:")
            print(f"  🔴 Live: {live_count}")
            print(f"  🟡 Upcoming: {upcoming_count}")
            print(f"  🟢 Completed: {completed_count}")
            
            if upcoming_count > 0:
                print(f"\n🟡 Sample upcoming match:")
                sample = data['upcoming'][0]
                print(f"  Name: {sample.get('name', 'N/A')}")
                print(f"  Teams: {sample.get('teams', [])}")
                print(f"  Date: {sample.get('date', 'N/A')}")
                print(f"  Status: {sample.get('status', 'N/A')}")
                
            if completed_count > 0:
                print(f"\n🟢 Sample completed match:")
                sample = data['completed'][0]
                print(f"  Name: {sample.get('name', 'N/A')}")
                print(f"  Teams: {sample.get('teams', [])}")
                print(f"  Status: {sample.get('status', 'N/A')}")
                
        else:
            print(f"❌ API endpoint failed: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_matches_endpoints()