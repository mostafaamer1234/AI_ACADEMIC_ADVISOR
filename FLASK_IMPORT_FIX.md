# 🔧 Flask Import Error Fix

## Problem
```
ModuleNotFoundError: No module named 'LLM'
```

This error occurs when Python can't find the `LLM` module because the Python path isn't set up correctly.

## ✅ Solutions Applied

### 1. **Fixed Import Order in api.py**
- Moved `sys.path.append()` BEFORE the LLM imports
- Added proper path setup before importing modules

### 2. **Created Dedicated Startup Script**
- **File:** `IAAS/LLMAcadamic-Advisor/start_flask.py`
- **Purpose:** Properly sets up Python path before starting Flask
- **Usage:** `python3 start_flask.py`

### 3. **Updated npm Scripts**
- Changed from `python3 -m api.api` to `python3 start_flask.py`
- Added better error handling

## 🚀 How to Start the Flask Server

### Option 1: Using npm (Recommended)
```bash
cd IAAS/client
npm start
```

### Option 2: Direct Python
```bash
cd IAAS/LLMAcadamic-Advisor
python3 start_flask.py
```

### Option 3: Manual with PYTHONPATH
```bash
cd IAAS/LLMAcadamic-Advisor
PYTHONPATH=. python3 api/api.py
```

## 🔍 Verification

After starting, you should see:
```
🚀 Starting Flask LLM Server...
📁 Working directory: /path/to/LLMAcadamic-Advisor
🐍 Python path: ['/path/to/LLMAcadamic-Advisor', ...]...
 * Serving Flask app 'api'
 * Debug mode: on
 * Running on http://127.0.0.1:5002
```

## 🐛 If Still Having Issues

### Check File Structure
```
IAAS/LLMAcadamic-Advisor/
├── api/
│   ├── __init__.py
│   └── api.py
├── LLM/
│   ├── main_copy.py
│   ├── majorsGraph.py
│   └── graphExperiment.py
└── start_flask.py
```

### Test Import Manually
```bash
cd IAAS/LLMAcadamic-Advisor
python3 -c "import sys; sys.path.insert(0, '.'); from LLM.main_copy import setup_environment; print('✅ Import successful')"
```

### Check Python Version
```bash
python3 --version
# Should be Python 3.8+
```

## 📋 Dependencies Check

Make sure all Python dependencies are installed:
```bash
pip3 install -r requirements.txt
```

## 🔄 Complete Reset

If nothing works:
```bash
# 1. Stop all servers
lsof -ti:3000,5001,5002 | xargs kill 2>/dev/null

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Test Flask server directly
cd IAAS/LLMAcadamic-Advisor
python3 start_flask.py

# 4. If that works, start everything
cd ../client
npm start
```

## ✅ Success Indicators

- Flask server starts without import errors
- Server runs on `http://127.0.0.1:5002`
- No `ModuleNotFoundError` messages
- API endpoints respond correctly
