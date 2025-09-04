import json
import re
import sys
import time

import requests

# Test configuration
user_id = "97d6e7df-f6d6-4eae-aa20-ef70b029a88f"

# Default test case
subject = "political"
question = "Có luận điệu cho rằng Việt Nam cần đa đảng như Mỹ mới là dân chủ và phát triển. Hãy phản biện lại luận điệu này."

# Allow command line arguments to override defaults
if len(sys.argv) > 1:
    subject = sys.argv[1]
if len(sys.argv) > 2:
    question = " ".join(sys.argv[2:])

# Baseline times for comparison
baseline_times = {
    "political": 33.72,  # From the issue description
    "legal": 136.28,     # From the issue description
    "default": 100.0     # Default baseline if not specified
}

baseline_time = baseline_times.get(subject, baseline_times["default"])
print(f"Running test with subject: {subject}")
print(f"Question: {question}")
print(f"Baseline time for comparison: {baseline_time:.2f} seconds\n")

payload = {
    "username": user_id,
    "subject": subject,
    "question": question
}

# Measure client-side time as well
start_time = time.time()
response = requests.post("http://127.0.0.1:8000/ask_bot", json=payload, headers={"User-Agent": "Gen Imagine Client"})
client_time = time.time() - start_time

try:
    response_json = response.json()

    # Print the response message
    print("Response:")
    print(response_json.get("message", "No message found"))

    # Print the response time if available from server
    if "response_time_seconds" in response_json:
        server_time = response_json["response_time_seconds"]
        print(f"\nServer Processing Time: {server_time:.2f} seconds")

    # Print client-side measured time
    print(f"Total Request Time: {client_time:.2f} seconds")

    # Check if the response contains references based on subject
    message = response_json.get("message", "")

    # Calculate improvement over baseline
    if "response_time_seconds" in response_json:
        server_time = response_json["response_time_seconds"]
        improvement = (baseline_time - server_time) / baseline_time * 100
        speedup = baseline_time / server_time if server_time > 0 else 0

        print(f"\nPerformance Improvement:")
        print(f"- {improvement:.2f}% faster than baseline")
        print(f"- {speedup:.2f}x speedup factor")

        if improvement >= 70:
            print("✓ TARGET ACHIEVED: Response time reduced by 70% or more")
        else:
            print(f"✗ TARGET NOT YET MET: Need {70-improvement:.2f}% more improvement")

    # Analyze response content
    print("\nResponse Analysis:")

    # Check response length
    word_count = len(message.split())
    print(f"- Word count: {word_count} words")
    if word_count < 50:
        print("✗ Response is too short (less than 50 words)")
    elif word_count > 200:
        print("✓ Response is detailed (more than 200 words)")

    if subject == "legal":
        # Check for legal document references
        legal_refs = []
        if "Luật" in message:
            legal_refs.append("Luật")
        if "Nghị định" in message:
            legal_refs.append("Nghị định")
        if "Thông tư" in message:
            legal_refs.append("Thông tư")
        if "Quyết định" in message:
            legal_refs.append("Quyết định")
        if "Điều" in message and re.search(r"Điều\s+\d+", message):
            legal_refs.append("Specific Articles")

        if legal_refs:
            print(f"✓ Legal references found: {', '.join(legal_refs)}")
        else:
            print("✗ No legal document references found")

    elif subject == "political":
        # Check for political document references
        political_refs = []
        if "Nghị quyết" in message:
            political_refs.append("Nghị quyết")
        if "Đại hội" in message:
            political_refs.append("Đại hội Đảng")
        if "Hội nghị Trung ương" in message:
            political_refs.append("Hội nghị Trung ương")
        if "Bác Hồ" in message or "Hồ Chí Minh" in message:
            political_refs.append("Hồ Chí Minh")
        if "Di chúc" in message:
            political_refs.append("Di chúc")

        if political_refs:
            print(f"✓ Political references found: {', '.join(political_refs)}")
        else:
            print("✗ No political document references found")

        # Check for specific Ho Chi Minh thought references
        hcm_specific_refs = re.findall(r"([\"'].*Hồ Chí Minh.*[\"'])", message)
        if hcm_specific_refs:
            print(f"✓ Specific Ho Chi Minh quotes/works found: {len(hcm_specific_refs)}")
            for ref in hcm_specific_refs[:3]:  # Show up to 3 examples
                print(f"  - {ref}")
        else:
            print("✗ No specific Ho Chi Minh quotes or works cited")

except json.JSONDecodeError:
    print("Error: Could not parse JSON response")
    print("Raw response:", response.text)
    print(f"Total Request Time: {client_time:.2f} seconds")
