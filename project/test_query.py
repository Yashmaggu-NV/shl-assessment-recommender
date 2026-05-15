"""
Quick test script to verify recommendation quality + refinement handling.
Run: python test_query.py
"""
import sys
import json
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()

from models.schemas import ChatRequest, Message
from agent.chat_logic import process_chat


def test_java_backend():
    """Test: Hiring mid-level Java backend engineer with AWS experience"""
    print("=" * 70)
    print("TEST 1: Initial recommendation")
    print("Query: Hiring mid-level Java backend engineer with AWS experience")
    print("=" * 70)

    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="Hiring mid-level Java backend engineer with AWS experience"
            )
        ]
    )

    response = process_chat(request)

    print(f"\nReply: {response.reply}\n")
    print(f"Recommendations ({len(response.recommendations)}):")
    for i, rec in enumerate(response.recommendations, 1):
        print(f"  [{i}] {rec.name} (type={rec.test_type})")
        print(f"      URL: {rec.url}")
    print(f"\nEnd of conversation: {response.end_of_conversation}")

    # Validate
    names = [r.name.lower() for r in response.recommendations]

    expected_keywords = ["java", "spring", "sql", "aws", "docker"]
    found = [kw for kw in expected_keywords if any(kw in n for n in names)]
    print(f"\n✓ Expected keywords found: {found}")

    bad_items = ["report", "360", "global skills", "virtual assessment",
                 "scenarios", "development center", "automata",
                 "verify", "deductive", "entry level", "j2ee",
                 "enterprise java beans", "java ee"]
    bad_found = [kw for kw in bad_items if any(kw in n for n in names)]
    if bad_found:
        print(f"✗ WARNING: Noise items leaked: {bad_found}")
    else:
        print("✓ No noise items in results")

    return response


def test_refinement(first_reply: str):
    """Test: Two-turn refinement — 'Add AWS' after initial recommendation"""
    print("\n" + "=" * 70)
    print("TEST 2: Refinement (Add AWS)")
    print("=" * 70)

    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="Hiring mid-level Java backend engineer with AWS experience"
            ),
            Message(
                role="assistant",
                content=first_reply,
            ),
            Message(
                role="user",
                content="Add AWS assessment"
            ),
        ]
    )

    response = process_chat(request)

    print(f"\nReply: {response.reply}\n")
    print(f"Recommendations ({len(response.recommendations)}):")
    for i, rec in enumerate(response.recommendations, 1):
        print(f"  [{i}] {rec.name} (type={rec.test_type})")

    # Should NOT be a clarification question
    lower_reply = response.reply.lower()
    if "what role" in lower_reply or "what position" in lower_reply:
        print("✗ FAIL: System asked for role on a refinement turn!")
    else:
        print("✓ System did NOT ask for role on refinement turn")

    if response.recommendations:
        print("✓ Recommendations returned (not empty)")
    else:
        print("✗ WARNING: No recommendations returned")

    return response


def test_removal(first_reply: str):
    """Test: Two-turn refinement — 'remove personality tests'"""
    print("\n" + "=" * 70)
    print("TEST 3: Refinement (Remove personality)")
    print("=" * 70)

    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="Hiring mid-level Java backend engineer with AWS experience"
            ),
            Message(
                role="assistant",
                content=first_reply,
            ),
            Message(
                role="user",
                content="remove personality tests"
            ),
        ]
    )

    response = process_chat(request)

    print(f"\nReply: {response.reply}\n")
    print(f"Recommendations ({len(response.recommendations)}):")
    for i, rec in enumerate(response.recommendations, 1):
        print(f"  [{i}] {rec.name} (type={rec.test_type})")

    lower_reply = response.reply.lower()
    if "what role" in lower_reply or "what position" in lower_reply:
        print("✗ FAIL: System asked for role on a refinement turn!")
    else:
        print("✓ System did NOT ask for role on refinement turn")

    return response


if __name__ == "__main__":
    r1 = test_java_backend()
    # Use the actual assistant reply for turn-2 tests
    first_reply = r1.reply
    test_refinement(first_reply)
    test_removal(first_reply)
