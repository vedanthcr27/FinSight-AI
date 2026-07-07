# 🚀 FinSight AI -- AI-Powered Stock Research Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Groq](https://img.shields.io/badge/Groq-LLM-green)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-blueviolet)
![License](https://img.shields.io/badge/License-MIT-yellow)

An AI-powered stock research platform that combines **real-time
financial market data** with **Large Language Models (LLMs)** to
generate structured investment insights through a modern Streamlit
dashboard.

> Built as a portfolio project for AI/ML, GenAI, and Software
> Engineering internship applications.

------------------------------------------------------------------------

# ✨ Features

-   📈 Real-time stock market data using yFinance
-   🤖 AI-powered investment analysis using Groq (Llama 3.1)
-   📊 Interactive Plotly price charts
-   💹 Financial metrics dashboard
-   🧠 Company overview, growth drivers, risks, bull/bear case
-   📑 AI investment summary and recommendation
-   ⚡ Modular architecture (UI, AI Engine, Data Layer)

------------------------------------------------------------------------

# 🖼️ Preview

Add screenshots after deployment.

    assets/
    ├── dashboard.png
    ├── analysis.png
    └── architecture.png

``` md
![Dashboard](assets/dashboard.png)

![Analysis](assets/analysis.png)
```

------------------------------------------------------------------------

# 🏗️ Architecture

``` text
             User
              │
              ▼
      Streamlit Dashboard
              │
              ▼
     Stock Search / Input
              │
              ▼
      yFinance Data API
              │
              ▼
      Financial Metrics
              │
              ▼
        Groq LLM Engine
              │
              ▼
     AI Investment Report
              │
              ▼
 Professional Interactive UI
```

------------------------------------------------------------------------

# 🛠 Tech Stack

  Category      Technologies
  ------------- ---------------------
  Language      Python
  Frontend      Streamlit
  Charts        Plotly
  AI            Groq API, Llama 3.1
  Finance       yFinance
  Data          Pandas, NumPy
  Environment   python-dotenv

------------------------------------------------------------------------

# 📂 Project Structure

``` text
FinSight-AI/
│
├── app.py
├── ai_engine.py
├── stock_data.py
├── search_stock.py
├── requirements.txt
├── .env.example
├── README.md
│
├── assets/
│   ├── dashboard.png
│   ├── analysis.png
│   └── architecture.png
│
└── screenshots/
```

------------------------------------------------------------------------

# 🚀 Installation

``` bash
git clone https://github.com/<YOUR_USERNAME>/FinSight-AI.git

cd FinSight-AI

python -m venv venv

source venv/bin/activate
```

Install dependencies

``` bash
pip install -r requirements.txt
```

Create `.env`

``` env
GROQ_API_KEY=YOUR_API_KEY
```

Run

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

# 📊 Current Features

-   Stock search
-   Real-time financial metrics
-   Interactive stock charts
-   AI stock analysis
-   Company overview
-   Growth drivers
-   Risk analysis
-   Bull case
-   Bear case
-   Investment summary

------------------------------------------------------------------------

# 📌 Future Roadmap

-   Company search autocomplete
-   News summarization
-   AI sentiment analysis
-   Portfolio tracker
-   Watchlist
-   Stock comparison
-   Earnings call summaries
-   Technical indicators (RSI, MACD)
-   Authentication
-   Cloud deployment

------------------------------------------------------------------------

# 📖 What I Learned

-   LLM API integration
-   Prompt engineering
-   Financial data analysis
-   Streamlit application development
-   Plotly visualizations
-   API error handling
-   Modular software architecture

------------------------------------------------------------------------

# 🌐 Live Demo

After deployment:

    https://your-app.streamlit.app

------------------------------------------------------------------------

# 💻 Resume Description

**FinSight AI -- AI-Powered Stock Research Assistant**

Developed an AI-powered stock research platform using Python, Streamlit,
Groq LLM, and yFinance that generates structured investment insights
from real-time market data. Built an interactive dashboard with
financial metrics, historical charts, and AI-generated analysis using a
modular architecture.

------------------------------------------------------------------------

# 🤝 Contributing

Contributions, ideas, and feature suggestions are welcome.

1.  Fork the repository
2.  Create a feature branch
3.  Commit your changes
4.  Open a Pull Request

------------------------------------------------------------------------

# 📄 License

MIT License

------------------------------------------------------------------------

## 👤 Author

**Vedanth C R**

-   GitHub: https://github.com/vedanthcr27
-   LinkedIn: https://www.linkedin.com/in/vedanth-cr-5ab29b329/

⭐ If you found this project useful, consider giving it a star.
