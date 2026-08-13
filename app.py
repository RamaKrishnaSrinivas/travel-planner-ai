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


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=os.getenv(
        "GEMINI_MODEL",
        "gemini-3.1-flash-lite"
    ),
    temperature=0.3
)


# ============================================================
# TOOLS
# ============================================================

search = DuckDuckGoSearchResults(
    num_results=5
)


geo = Nominatim(
    user_agent="travel-planner-ai",
    timeout=10
)


# ============================================================
# DISTANCE TOOL
# ============================================================

@tool
def distance(origin: str, destination: str) -> str:
    """Calculate approximate distance between two places in kilometers."""

    try:

        a = geo.geocode(
            origin,
            timeout=10
        )

        b = geo.geocode(
            destination,
            timeout=10
        )

        if not a or not b:
            return "Distance unavailable: location not found."

        km = geodesic(
            (a.latitude, a.longitude),
            (b.latitude, b.longitude)
        ).km

        return f"Approximate distance: {km:.0f} km"

    except Exception:
        return "Distance information is temporarily unavailable."


# ============================================================
# WEATHER TOOL
# ============================================================

@tool
def weather(city: str) -> str:
    """Get current weather information for a city."""

    try:

        location = geo.geocode(
            city,
            timeout=10
        )

        if not location:
            return "Weather unavailable: city not found."

        url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m"
            ),
            "timezone": "auto"
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if "current" not in data:
            return "Weather information is temporarily unavailable."

        current = data["current"]

        temperature = current.get(
            "temperature_2m"
        )

        humidity = current.get(
            "relative_humidity_2m"
        )

        return (
            f"Current weather in {city}: "
            f"{temperature}°C, "
            f"humidity {humidity}%."
        )

    except Exception:
        return "Weather information is temporarily unavailable."


# ============================================================
# BUDGET CALCULATOR
# ============================================================

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

    try:

        total = (
            hotel
            + food
            + transport
            + activities
        ) * days * people

        return (
            f"Estimated budget: "
            f"₹{total:,.0f}"
        )

    except Exception:
        return "Budget calculation is unavailable."


# ============================================================
# TOOL LIST
# ============================================================

tools = [
    search,
    distance,
    weather,
    calculator
]


# ============================================================
# AI AGENT
# ============================================================

prompt = """
You are a Travel Planner AI Agent.

Create a useful and beginner-friendly travel plan.

For every travel request try to provide:

1. Trip Summary
2. Distance
3. Current Weather
4. Five Popular Places
5. Day-by-Day Itinerary
6. Estimated Budget
7. Travel Tips

TOOL RULES:

- Use web search for popular places.
- Use the distance tool for travel distance.
- Use the weather tool for current weather.
- Use the calculator for estimated budget.

If a tool is temporarily unavailable,
DO NOT stop the entire travel plan.

Instead, continue the plan and write:

"Information temporarily unavailable."

For Indian trips, use Indian Rupees.

Return ONLY clean, readable text.

Do NOT show:

- JSON
- dictionaries
- metadata
- tool calls
- function calls
- internal information
- Python code

Use simple headings and bullet points.

Keep the answer useful and beginner-friendly.
"""


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt
)


# ============================================================
# TRAVEL FUNCTION
# ============================================================

def travel(question: str) -> str:

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            }
        )

        messages = result.get(
            "messages",
            []
        )

        for message in reversed(messages):

            if message.__class__.__name__ != "AIMessage":
                continue

            content = message.content

            # Normal text response
            if isinstance(content, str):

                if content.strip():
                    return content.strip()

            # Gemini sometimes returns a list
            if isinstance(content, list):

                text = []

                for item in content:

                    if isinstance(item, str):
                        text.append(item)

                    elif isinstance(item, dict):

                        if item.get("type") == "text":

                            value = item.get(
                                "text",
                                ""
                            )

                            if value:
                                text.append(value)

                answer = "\n".join(
                    text
                ).strip()

                if answer:
                    return answer

        return "No travel plan was generated."

    except Exception as e:

        print(
            "Travel planning error:",
            repr(e)
        )

        return (
            "Sorry, I could not create the "
            "travel plan right now. "
            "Please try again in a few seconds."
        )


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>Travel Planner AI</title>

<meta name="viewport"
      content="width=device-width, initial-scale=1">

<style>

body {
    font-family: Arial, sans-serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 20px;
    background: #f5f5f5;
    color: #222;
}

h1 {
    text-align: center;
}

.description {
    text-align: center;
    color: #555;
}

textarea {
    width: 100%;
    height: 120px;
    padding: 12px;
    font-size: 16px;
    box-sizing: border-box;
    border: 1px solid #ccc;
    border-radius: 6px;
    resize: vertical;
}

button {
    margin-top: 12px;
    padding: 12px 25px;
    font-size: 16px;
    cursor: pointer;
    border: none;
    border-radius: 6px;
    background: #222;
    color: white;
}

button:hover {
    background: #444;
}

.result {
    margin-top: 30px;
    padding: 25px;
    background: white;
    border-radius: 8px;
    line-height: 1.6;
    white-space: pre-wrap;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

</style>

</head>


<body>

<h1>✈️ Travel Planner AI</h1>

<p class="description">
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


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    answer = ""
    question = ""

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        if question:

            answer = travel(
                question
            )

    return render_template_string(
        HTML,
        answer=answer,
        question=question
    )


# ============================================================
# LOCAL / RENDER SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )
