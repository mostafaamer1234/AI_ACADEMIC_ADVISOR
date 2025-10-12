# 🚀 AWS Deployment Fix - File Path Issues

## Problem
```
FileNotFoundError: [Errno 2] No such file or directory: 'AI_ACADEMIC_ADVISOR/IAAS/LLMAcadamic-Advisor/LLM/processed_psu_courses.csv'
```

This error occurs because the code uses hardcoded absolute paths that only work on the local development machine.

## ✅ Solutions Applied

### 1. **Fixed All Hardcoded Paths**
Updated all files to use relative paths that work on any system:

**Files Updated:**
- `LLM/main.py` - Fixed CSV path
- `LLM/main_copy.py` - Fixed CSV path  
- `LLM/graphExperiment.py` - Fixed CSV path
- `LLM/majorsGraph.py` - Fixed CSV paths (2 instances)
- `api/api.py` - Fixed CSV path

### 2. **Path Changes Made**

**Before (Hardcoded):**
```python
csv_path = "/Users/mostafa/newFianlLLMRepo/IAAS/LLMAcadamic-Advisor/LLM/processed_psu_courses.csv"
```

**After (Dynamic):**
```python
csv_path = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
```

### 3. **Files That Need to Exist**

Make sure these CSV files are in the `LLM/` directory:
- ✅ `processed_psu_courses.csv`
- ✅ `all_abington_majors_combined(5).csv`

## 🔧 AWS Deployment Steps

### 1. **Upload All Files**
Make sure to upload the entire project structure to AWS, including:
```
AI_ACADEMIC_ADVISOR/
├── IAAS/
│   └── LLMAcadamic-Advisor/
│       ├── LLM/
│       │   ├── processed_psu_courses.csv          # ← REQUIRED
│       │   ├── all_abington_majors_combined(5).csv # ← REQUIRED
│       │   ├── main.py
│       │   ├── main_copy.py
│       │   ├── graphExperiment.py
│       │   └── majorsGraph.py
│       └── api/
│           └── api.py
```

### 2. **Install Dependencies**
```bash
cd AI_ACADEMIC_ADVISOR
pip3 install -r requirements.txt
```

### 3. **Test the Fix**
```bash
cd IAAS/LLMAcadamic-Advisor
python3 start_flask.py
```

### 4. **Verify Files Exist**
```bash
ls -la LLM/
# Should show:
# - processed_psu_courses.csv
# - all_abington_majors_combined(5).csv
```

## 🐛 Troubleshooting

### If Still Getting File Not Found Errors:

1. **Check File Locations:**
   ```bash
   find . -name "*.csv" -type f
   ```

2. **Check Current Directory:**
   ```bash
   pwd
   ls -la
   ```

3. **Test Path Resolution:**
   ```python
   import os
   print(os.path.dirname(__file__))
   print(os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv"))
   ```

### Common Issues:

1. **Missing CSV Files:** Upload the CSV files to the correct location
2. **Wrong Working Directory:** Run from the correct directory
3. **Permission Issues:** Check file permissions
4. **Path Separators:** Use `os.path.join()` for cross-platform compatibility

## ✅ Success Indicators

- Flask server starts without FileNotFoundError
- CSV files are found and loaded successfully
- No hardcoded paths in error messages
- Application works on both local and AWS environments

## 📋 Pre-Deployment Checklist

- [ ] All CSV files uploaded to correct locations
- [ ] No hardcoded paths in code
- [ ] Dependencies installed (`pip3 install -r requirements.txt`)
- [ ] Test locally first
- [ ] Verify file permissions on AWS
- [ ] Check working directory when running

## 🚀 Quick Fix Commands

```bash
# 1. Check if files exist
ls -la IAAS/LLMAcadamic-Advisor/LLM/*.csv

# 2. Test import
cd IAAS/LLMAcadamic-Advisor
python3 -c "import os; print(os.path.join(os.path.dirname('LLM/main.py'), 'processed_psu_courses.csv'))"

# 3. Start Flask server
python3 start_flask.py
```

The application should now work on AWS Ubuntu without file path errors!
