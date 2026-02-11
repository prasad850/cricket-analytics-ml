#!/usr/bin/env python3
# test_matches.py - Test live matches functionality

import sys
import os
sys.path.append(os.path.dirname(__file__))

from a import fetch_cricapi_matches, make_session, process_matches

def test_matches():
    print("🏏 Testing Live Matches Functionality...")
    
    # Create session
    session = make_session()
    
    # Fetch Cricket API data
    print("📡 Fetching Cricket API data...")
    cricapi_data = fetch_cricapi_matches(session)
    
    if not cricapi_data:
        print("❌ No Cricket API data received")
        return
    
    print(f"✅ Received data with {len(cricapi_data.get('data', []))} matches")
    
    # Process matches
    combined = {'cricapi': cricapi_data}
    matches = process_matches(combined)
    
    print(f"\n📊 Match Categories:")
    print(f"🔴 Live matches: {len(matches['live'])}")
    print(f"🟡 Upcoming matches: {len(matches['upcoming'])}")
    print(f"🟢 Completed matches: {len(matches['completed'])}")
    
    print(f"\n🎯 Sample match statuses:")
    if cricapi_data and 'data' in cricapi_data:
        for i, match in enumerate(cricapi_data['data'][:5]):
            name = match.get('name', 'Unknown')[:50]
            status = match.get('status', 'No status')
            print(f"  {i+1}. {name}: {status}")
    
    # Show live matches specifically
    if matches['live']:
        print(f"\n🔴 Live Matches Found:")
        for match in matches['live']:
            print(f"  - {match['name']}: {match['status']}")
    
    if matches['upcoming']:
        print(f"\n🟡 Upcoming Matches Found:")
        for match in matches['upcoming'][:3]:
            print(f"  - {match['name']}: {match['status']}")

if __name__ == "__main__":
    test_matches()