# 🎓 AI Academic Advisor System (IAAS)

An intelligent academic advising system that helps students understand their academic progress, generate personalized graduation plans, and explore course prerequisites using advanced AI and machine learning.

## 🌟 Features

- **🔐 User Authentication**: Secure login system for students, faculty, and administrators
- **📄 PDF Transcript Processing**: Upload and analyze academic transcripts using AI
- **🎯 Graduation Planning**: Generate personalized semester-by-semester graduation plans
- **🤖 AI Chat Advisor**: Interactive chatbot for academic guidance
- **📊 Course Prerequisites**: Visual graph-based course prerequisite analysis
- **👥 Multi-Role Support**: Student, Faculty, and Admin dashboards

## 🚀 Quick Start

### Prerequisites
- Node.js (v16+)
- Python 3.8+
- MySQL/XAMPP
- Git

### Python Dependencies
Install Python dependencies using one of these methods:

**Option 1: Automated Installation**
```bash
./install_python_deps.sh
```

**Option 2: Manual Installation**
```bash
pip3 install -r requirements.txt
```

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mostafaamer1234/AI_ACADEMIC_ADVISOR.git
   cd AI_ACADEMIC_ADVISOR/IAAS/client
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

3. **Start the application:**
   ```bash
   npm start
   ```

That's it! The application will start all three servers automatically:
- 🔵 **Frontend**: http://localhost:3000
- 🟢 **Backend**: http://localhost:5001
- 🟣 **LLM Server**: http://localhost:5002

## 📖 Documentation

- **[Quick Start Guide](QUICK_START.md)** - One-page setup guide
- **[Startup Guide](IAAS/client/STARTUP.md)** - Detailed setup instructions
- **[Troubleshooting](IAAS/client/TROUBLESHOOTING.md)** - Complete troubleshooting guide

## 🏗️ Architecture

### Frontend (React)
- **Location**: `IAAS/client/`
- **Port**: 3000
- **Features**: User interface, file upload, chat interface

### Backend (Node.js + Express)
- **Location**: `IAAS/client/server/`
- **Port**: 5001
- **Features**: Authentication, database management, user management

### LLM Server (Flask + Python)
- **Location**: `IAAS/LLMAcadamic-Advisor/api/`
- **Port**: 5002
- **Features**: PDF processing, AI graduation planning, course analysis

### Database
- **Type**: MySQL
- **Database**: `Advising`
- **Tables**: `users`, course data, academic records

## 🛠️ Technology Stack

### Frontend
- React 18
- Axios for API calls
- Material-UI components
- React Router for navigation

### Backend
- Node.js + Express
- MySQL with connection pooling
- bcrypt for password hashing
- CORS enabled

### AI/ML
- Python Flask API
- LangChain for document processing
- OpenAI GPT-4 for AI responses
- ChromaDB for vector storage
- PyPDF for PDF processing

## 📁 Project Structure

```
AI_ACADEMIC_ADVISOR/
├── IAAS/
│   ├── client/                 # React frontend
│   │   ├── src/
│   │   │   ├── components/     # React components
│   │   │   ├── pages/         # Page components
│   │   │   ├── services/      # API services
│   │   │   └── styles/        # CSS styles
│   │   ├── server/            # Node.js backend
│   │   ├── setup.sh           # Setup script
│   │   ├── STARTUP.md         # Startup guide
│   │   └── TROUBLESHOOTING.md # Troubleshooting guide
│   └── LLMAcadamic-Advisor/   # AI/ML backend
│       ├── api/               # Flask API
│       ├── LLM/               # AI processing modules
│       └── requirements.txt   # Python dependencies
├── chroma_courses/            # Vector database
├── requirements.txt           # Main Python dependencies
├── install_python_deps.sh     # Python setup script
├── QUICK_START.md            # Quick start guide
└── README.md                 # This file
```

## 🔧 Configuration

### Environment Variables
The application uses the following environment variables (configured in the code):

- `LANGCHAIN_API_KEY`: LangChain API key
- `OPENAI_API_KEY`: OpenAI API key
- `LANGCHAIN_ENDPOINT`: LangChain endpoint URL

### Database Setup
1. Start MySQL/XAMPP
2. Create database named `Advising`
3. Import user tables and course data

## 🚨 Troubleshooting

### Common Issues

1. **Login fails**: Check if backend server is running on port 5001
2. **PDF upload fails**: Check if LLM server is running on port 5002
3. **CORS errors**: Restart all servers with `npm start`
4. **Python not found**: Use `python3` (already configured)

### Complete Reset
```bash
# Stop all servers
lsof -ti:3000,5001,5002 | xargs kill 2>/dev/null

# Restart everything
cd IAAS/client
npm start
```

For detailed troubleshooting, see [TROUBLESHOOTING.md](IAAS/client/TROUBLESHOOTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is private and proprietary. All rights reserved.

## 👨‍💻 Author

**Mostafa Amer**
- GitHub: [@mostafaamer1234](https://github.com/mostafaamer1234)
- Repository: https://github.com/mostafaamer1234/AI_ACADEMIC_ADVISOR

## 🎯 Features in Detail

### User Authentication
- Secure login system with bcrypt password hashing
- Role-based access (Student, Faculty, Admin)
- Session management with JWT tokens

### PDF Processing
- Upload academic transcripts in PDF format
- AI-powered document analysis and extraction
- Student information redaction for privacy

### Graduation Planning
- AI-generated personalized graduation plans
- Semester-by-semester course recommendations
- Prerequisite checking and validation
- Credit hour calculations

### AI Chat Advisor
- Interactive chatbot for academic questions
- Context-aware responses based on student data
- Course recommendation and planning assistance

## 🔒 Security Features

- Password hashing with bcrypt
- CORS protection
- Input validation and sanitization
- Secure file upload handling
- Error handling without information leakage

## 📊 Performance

- Concurrent server management
- Efficient database queries
- Vector database for fast AI processing
- Optimized React components
- Background processing for heavy tasks

---

**🎉 Ready to use!** The application is production-ready and includes comprehensive documentation, automated setup, and troubleshooting guides.