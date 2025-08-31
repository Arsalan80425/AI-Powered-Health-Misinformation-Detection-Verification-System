# 🏥 AI Health Misinformation Detection  
**Advanced Medical Fact-Checking System (DataLEADS FirstCheck Initiative)**  

Visit the live demo at : https://arsalan80425-ai-powered-health-misinformation-detec-main-obq7u1.streamlit.app

## 📌 Overview  
This project is an **AI-powered health misinformation detection system** built with **Streamlit**.  
It analyzes health-related claims, checks them against **PubMed research papers, trusted medical sources (WHO, CDC, NIH)**, and a **custom knowledge base (RAG + FAISS)**, then generates professional fact-check reports with risk assessments.  

The system is aligned with **DataLEADS FirstCheck Initiative**, aiming to combat medical misinformation with automation and AI.  

---

## 🖼️ Dashboard Preview

![Dashboard Preview 1](https://raw.githubusercontent.com/Arsalan80425/AI-Powered-Health-Misinformation-Detection-Verification-System/refs/heads/master/preview1.png)
![Dashboard Preview 2](https://raw.githubusercontent.com/Arsalan80425/AI-Powered-Health-Misinformation-Detection-Verification-System/refs/heads/master/preview2.png)
![Dashboard Preview 3](https://raw.githubusercontent.com/Arsalan80425/AI-Powered-Health-Misinformation-Detection-Verification-System/refs/heads/master/preview3.png)
![Dashboard Preview 4](https://raw.githubusercontent.com/Arsalan80425/AI-Powered-Health-Misinformation-Detection-Verification-System/refs/heads/master/preview4.png)
![Dashboard Preview 5](https://raw.githubusercontent.com/Arsalan80425/AI-Powered-Health-Misinformation-Detection-Verification-System/refs/heads/master/preview5.png)

---

## 🚀 Features  
✅ **AI-Powered Claim Verification**  
- BioBERT & SciBERT for classification  
- Zero-shot classification (BART-Large-MNLI)  
- Medical NER for entity extraction  

✅ **Real-Time Evidence Retrieval**  
- Live PubMed API integration  
- Trusted health authority sources (WHO, CDC, NIH)  
- RAG-powered knowledge base with FAISS  

✅ **Risk Assessment & Analytics**  
- Confidence calibration & misinformation danger scoring  
- Interactive analytics dashboard with trends, entity analysis, and distributions  

✅ **Professional Reporting**  
- Comprehensive fact-check reports  
- JSON/Markdown/CSV export options  
- Batch claim analysis for professionals  

---

## 🛠️ Installation  

1. **Clone the repository**  
```bash
git clone https://github.com/Arsalan80425/AI-Powered-Health-Misinformation-Detection-Verification-System.git
cd AI-Powered-Health-Misinformation-Detection-Verification-System
```

2. **Create and activate virtual environment**  
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

*(Make sure you have Python 3.9+ installed)*  

---

## ▶️ Usage  

Run the Streamlit app:  
```bash
streamlit run main.py
```

Then open the app in your browser at:  
👉 http://localhost:8501  

---

## 📊 Analytics Dashboard  
The app includes a **built-in analytics dashboard** that provides:  
- Claim classification distribution (pie & bar charts)  
- Confidence & risk score distributions  
- Top medical entities identified  
- Fact-checks over time (trend analysis)  
- Date-range filtering & CSV export  

---

## 📂 Project Structure  
```
.
├── main.py              # Streamlit app with AI fact-checking pipeline
├── enhanced_factcheck.db # SQLite database (auto-created on first run)
├── health_knowledge_base.pkl # Knowledge base (auto-generated)
├── health_vectors.index  # FAISS index (auto-generated)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 📚 Tech Stack  
- **Frontend/UI**: Streamlit  
- **NLP Models**: BioBERT, SciBERT, MiniLM, BART-Large-MNLI  
- **Evidence Retrieval**: PubMed API, WHO/CDC/NIH guidelines  
- **Vector Database**: FAISS + Sentence Transformers  
- **Database**: SQLite (with caching & analytics)  
- **Visualization**: Plotly, Streamlit charts  

---

## 🏆 Achievements  
- Real-time PubMed research integration  
- Multi-source medical claim verification  
- Risk scoring for misinformation detection  
- Analytics & reporting for professionals  
- Production-ready architecture with caching & RAG pipelines  

---

## ⚠️ Disclaimer  
This system is **for educational and research purposes only**.  
Always consult **qualified healthcare professionals** for actual medical advice.  


---

### Contact Information
- **Developer**: Mohammed Arsalan
- **Email**: arsalanshaikh0408@gmail.com
- **LinkedIn**: [[LinkedIn Profile](http://www.linkedin.com/in/mohammed-arsalan-58543a305)]
- **Project**: Advanced Medical Fact-Checking System (DataLEADS FirstCheck Initiative)

---

## 👨‍💻 Author  
Developed by **Mohammed Arsalan**  
🎯 Built for **DataLEADS FirstCheck Initiative**  

