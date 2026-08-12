import os
import requests
from flask import Flask, request, render_template_string
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

# -----------------------------
# Gemini
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
)

# -----------------------------
# Tools
# -----------------------------

search = DuckDuckGoSearchResults(num_results=5)

geo = Nominatim(user_agent="travel-planner-ai")


@tool
def distance(origin: str, destination: str) -> str:
    """Calculate distance between two places in km."""

    a = geo.geocode(origin)
    b = geo.geocode(destination)

    if not a or not b:
        return "Location not found."

    km = geodesic(
        (a.latitude, a.longitude),
        (b.latitude, b.longitude)
    ).km

    return f"Approximate distance: {km:.0f} km"


@tool
def weather(city: str) -> str:
    """Get current weather of a city."""

    location = geo.geocode(city)

    if not location:
        return "City not found."

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={location.latitude}"
        f"&longitude={location.longitude}"
        f"&current=temperature_2m,relative_humidity_2m"
    )

    data = requests.get(url, timeout=10).json()["current"]

    return (
        f"Temperature: {data['temperature_2m']}°C, "
        f"Humidity: {data['relative_humidity_2m']}%"
    )


@tool
def calculator(
    hotel: float,
    food: float,
    transport: float,
    activities: float,
    days: int,
    people: int
) -> str:
    """Calculate estimated travel budget in Indian Rupees."""

    total = (
        hotel
        + food
        + transport
        + activities
    ) * days * people

    return f"Estimated budget: ₹{total:,.0f}"


tools = [
    search,
    distance,
    weather,
    calculator
]

# -----------------------------
# Agent
# -----------------------------

prompt = """
You are a Travel Planner AI Agent.

For every travel request provide:

1. Trip summary
2. Distance
3. Current weather
4. Five popular places
5. Day-by-day itinerary
6. Estimated budget
7. Travel tips

Use the available tools.

Use web search for popular places.
Use distance for travel distance.
Use weather for current weather.
Use calculator for budget.

For Indian trips use Indian Rupees.

Return ONLY clean, readable text.
Do not show JSON, dictionaries, metadata,
tool calls or internal information.

Keep the answer useful and beginner-friendly.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt
)


# -----------------------------
# Travel function
# -----------------------------

def travel(question):

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    for message in reversed(result["messages"]):

        if message.__class__.__name__ != "AIMessage":
            continue

        content = message.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):

            text = []

            for item in content:

                if isinstance(item, str):
                    text.append(item)

                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        text.append(item.get("text", ""))

            answer = "\n".join(text).strip()

            if answer:
                return answer

    return "No travel plan was generated."


# -----------------------------
# Flask Web App
# -----------------------------

app = Flask(__name__)


HTML = """
<!DOCTYPE html>
<html>

<head>

<title>Travel Planner AI</title>

<style>

body {
    font-family: Arial;
    max-width: 900px;
    margin: 40px auto;
    padding: 20px;
    background: #f5f5f5;
}

h1 {
    text-align: center;
}

textarea {
    width: 100%;
    height: 100px;
    padding: 10px;
    font-size: 16px;
}

button {
    margin-top: 10px;
    padding: 12px 25px;
    font-size: 16px;
    cursor: pointer;
}

.result {
    margin-top: 30px;
    padding: 20px;
    background: white;
    border-radius: 8px;
    white-space: pre-wrap;
}

</style>

</head>

<body>

<h1>✈️ Travel Planner AI</h1>

<p>
Ask the AI to create your travel plan.
</p>

<form method="POST">

<textarea
name="question"
placeholder="Example: Plan a 5-day trip from Bangalore to Goa for 2 people."
required
>{{ question }}</textarea>

<br>

<button type="submit">
Create Travel Plan
</button>

</form>

{% if answer %}

<div class="result">

{{ answer }}

</div>

{% endif %}

</body>

</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    answer = ""
    question = ""

    if request.method == "POST":

        question = request.form.get("question", "")

        if question:
            answer = travel(question)

    return render_template_string(
        HTML,
        answer=answer,
        question=question
    )


# -----------------------------
# Local testing
# -----------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )