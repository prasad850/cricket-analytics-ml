# test_cricket_api.py
# Quick test script to verify your Cricket API setup

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_cricket_api():
    """Test the Cricket API connection"""
    
    cricapi_url = os.getenv("CRICAPI_URL", "").strip()
    cricapi_key = os.getenv("CRICAPI_KEY", "").strip()
    
    print("🏏 Testing Cricket API Configuration...")
    print(f"URL: {cricapi_url}")
    print(f"API Key: {'✅ Configured' if cricapi_key else '❌ Missing'}")
    
    if not cricapi_url or not cricapi_key:
        print("\n❌ Missing API configuration!")
        print("Please update your .env file with:")
        print("CRICAPI_URL=https://api.cricapi.com/v1/matches")
        print("CRICAPI_KEY=your_actual_api_key_here")
        return
    
    try:
        # Test API call
        headers = {"X-API-KEY": cricapi_key} if cricapi_key else {}
        
        print(f"\n🔄 Testing API call to: {cricapi_url}")
        response = requests.get(cricapi_url, headers=headers, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Test Successful!")
            
            if 'data' in data and len(data['data']) > 0:
                print(f"📊 Found {len(data['data'])} matches")
                print(f"Sample match: {data['data'][0].get('name', 'Unknown')}")
            else:
                print("📊 API connected but no matches found")
                
        elif response.status_code == 401:
            print("❌ Authentication failed! Check your API key")
        elif response.status_code == 429:
            print("⚠️ Rate limit exceeded! Try again later")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        print("Check your internet connection and API URL")
    
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_cricket_api()