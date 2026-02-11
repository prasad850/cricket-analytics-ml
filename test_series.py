from a import fetch_cricapi_matches, process_matches, extract_series_name
import requests

def test_series_categorization():
    """Test the new series categorization functionality"""
    print("🏆 Testing Series Categorization")
    print("=" * 40)
    
    try:
        # Get Cricket API data
        print("📡 Fetching Cricket API data...")
        session = requests.Session()
        raw_data = fetch_cricapi_matches(session)
        
        if raw_data:
            print(f"✅ Received {len(raw_data.get('data', []))} matches")
            
            # Format data for processing
            formatted_data = {"cricapi": raw_data}
            
            # Process matches by series
            series_matches = process_matches(formatted_data)
            
            print(f"\n🎯 Found {len(series_matches)} series:")
            
            for series_name, matches in series_matches.items():
                print(f"\n🏏 {series_name} ({len(matches)} matches)")
                
                # Count match types in this series
                live_count = sum(1 for m in matches if m.get('match_status') == 'live')
                upcoming_count = sum(1 for m in matches if m.get('match_status') == 'upcoming')
                completed_count = sum(1 for m in matches if m.get('match_status') == 'completed')
                
                print(f"   🔴 Live: {live_count} | 🟡 Upcoming: {upcoming_count} | 🟢 Completed: {completed_count}")
                
                # Show sample matches
                for i, match in enumerate(matches[:2]):  # Show first 2 matches
                    status_emoji = {"live": "🔴", "upcoming": "🟡", "completed": "🟢"}.get(match.get('match_status', ''), "⚪")
                    print(f"   {status_emoji} {match.get('teams', ['TBD', 'TBD'])}")
            
            print(f"\n✅ Series categorization working successfully!")
            print(f"Total matches processed: {sum(len(matches) for matches in series_matches.values())}")
            
        else:
            print("❌ Failed to fetch Cricket API data")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_series_extraction():
    """Test the series name extraction function"""
    print("\n🔍 Testing Series Name Extraction")
    print("=" * 40)
    
    test_cases = [
        "England U19 vs India U19, Final, ICC Under 19 World Cup 2026",
        "Madhya Pradesh vs Jammu and Kashmir, 2nd Quarter-Final (B1 v D2), Ranji Trophy Elite 2025-26",
        "Pakistan A vs England Lions, 1st T20, Pakistan A v England Lions in UAE, 2026",
        "Tbc vs Tbc, Final, The Hundred Mens Competition 2026"
    ]
    
    for match_name in test_cases:
        series = extract_series_name(match_name)
        print(f"Match: {match_name}")
        print(f"Series: {series}")
        print()

if __name__ == "__main__":
    test_series_extraction()
    test_series_categorization()