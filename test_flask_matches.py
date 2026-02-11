import requests
import json

def test_flask_matches():
    """Test the Flask app's matches endpoint"""
    try:
        print("🌐 Testing Flask /matches endpoint...")
        
        # Test the matches endpoint
        response = requests.get("http://127.0.0.1:5000/api/matches", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Flask matches API working")
            print(f"📊 Matches data received:")
            
            # Count matches by type
            live_count = len(data.get('live', []))
            upcoming_count = len(data.get('upcoming', []))
            completed_count = len(data.get('completed', []))
            
            print(f"🔴 Live matches: {live_count}")
            print(f"🟡 Upcoming matches: {upcoming_count}")
            print(f"🟢 Completed matches: {completed_count}")
            
            # Show sample upcoming matches
            if upcoming_count > 0:
                print("\n🟡 Sample upcoming matches:")
                for i, match in enumerate(data['upcoming'][:3]):
                    print(f"  {i+1}. {match.get('teams', ['TBD', 'TBD'])} - {match.get('name', 'Unknown')}")
            
            # Show sample completed matches  
            if completed_count > 0:
                print("\n🟢 Sample completed matches:")
                for i, match in enumerate(data['completed'][:3]):
                    print(f"  {i+1}. {match.get('teams', ['TBD', 'TBD'])} - {match.get('status', 'No result')}")
                    
        else:
            print(f"❌ Flask matches API error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing Flask endpoint: {e}")

if __name__ == "__main__":
    test_flask_matches()