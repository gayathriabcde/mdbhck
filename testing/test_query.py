import requests
import json

# Your local Flask API URL
API_URL = "http://localhost:5000/api/query"


def test_retrieval(user_query):
    print(f"\n🔍 Searching for: '{user_query}'...")

    payload = {
        "query": user_query,
        "limit": 1  # We only want the top result for this test
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        if data['results_count'] == 0:
            print("❌ No matching nodes found.")
            return

        # Extract the top result
        top_result = data['data'][0]
        target_time = top_result['timestamp']

        print("\n" + "=" * 50)
        print("🎯 TARGET NODE FOUND")
        print("=" * 50)
        print(f"Timestamp:   {target_time}s")
        print(f"Visual:      {top_result['visual_caption']}")
        print(f"Audio:       {top_result['audio_transcript']}")
        print(f"Dissonance:  {top_result.get('dissonance_score', 'N/A')}")

        print("\n" + "-" * 50)
        print("⏪ PREREQUISITE STEPS (Graph Traversal)")
        print("-" * 50)

        prereqs = top_result.get('prerequisite_steps', [])
        if not prereqs:
            print("No previous steps found (This might be the start of the video).")
        else:
            # Sort by steps_backward descending so it reads chronologically
            prereqs_sorted = sorted(prereqs, key=lambda x: x['steps_backward'], reverse=True)
            for step in prereqs_sorted:
                depth = step['steps_backward']
                print(f"[T - {depth + 1} steps] at {step['timestamp']}s:")
                print(f"   👁️ Visual: {step['visual_caption']}")
                print(f"   🔊 Audio:  {step['audio_transcript']}\n")

    except Exception as e:
        print(f"❌ Error pinging API: {e}")


if __name__ == "__main__":
    # Test a few different queries!
    queries_to_test = [
        "tightening the bolt",
        "inserting the wire",
        "cutting the wood",
        "How do I fold the wings?"
    ]

    for q in queries_to_test:
        test_retrieval(q)
        input("\nPress Enter to test next query...")