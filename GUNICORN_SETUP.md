# 🚀 Gunicorn Setup for AWS Deployment

## Problem
```
ModuleNotFoundError: No module named 'app'
```

This happens because Gunicorn is looking for `app:app` but can't find the Flask app module.

## ✅ Solution

### 1. **Navigate to the Correct Directory**
```bash
cd ~/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/LLMAcadamic-Advisor
```

### 2. **Use the Correct Module Path**
```bash
# Instead of: gunicorn -w 3 -b 0.0.0.0:5002 app:app
# Use this:
gunicorn -w 3 -b 0.0.0.0:5002 api.api:app
```

### 3. **Set Python Path (Important)**
```bash
# Set PYTHONPATH to current directory
export PYTHONPATH=.
gunicorn -w 3 -b 0.0.0.0:5002 api.api:app
```

### 4. **Complete Command Sequence**
```bash
cd ~/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/LLMAcadamic-Advisor
export PYTHONPATH=.
gunicorn -w 3 -b 0.0.0.0:5002 api.api:app
```

## 🔧 Alternative: Use the Startup Script

### Option 1: Use the startup script we created
```bash
cd ~/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/LLMAcadamic-Advisor
python3 start_flask.py
```

### Option 2: Create a Gunicorn config file
Create `gunicorn.conf.py`:
```python
bind = "0.0.0.0:5002"
workers = 3
worker_class = "sync"
pythonpath = "."
```

Then run:
```bash
gunicorn -c gunicorn.conf.py api.api:app
```

## 🐛 Troubleshooting

### If still getting import errors:
```bash
# Check if you're in the right directory
pwd
# Should show: /home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/LLMAcadamic-Advisor

# Check if api directory exists
ls -la api/

# Check if api.py exists
ls -la api/api.py

# Test import manually
python3 -c "import api.api; print('✅ Import successful')"
```

### Check file structure:
```
IAAS/LLMAcadamic-Advisor/
├── api/
│   ├── __init__.py
│   └── api.py          # ← This contains the Flask app
├── LLM/
│   ├── main_copy.py
│   └── ...
└── start_flask.py
```

## 🚀 Production Setup

### Create a systemd service (recommended):
Create `/etc/systemd/system/flask-app.service`:
```ini
[Unit]
Description=Flask App
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/LLMAcadamic-Advisor
Environment=PATH=/home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/venv/bin
Environment=PYTHONPATH=/home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/LLMAcadamic-Advisor
ExecStart=/home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/venv/bin/gunicorn -w 3 -b 0.0.0.0:5002 api.api:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable flask-app
sudo systemctl start flask-app
sudo systemctl status flask-app
```

## ✅ Expected Success Output

When it works, you should see:
```
[2025-10-12 21:30:00 +0000] [5835] [INFO] Starting gunicorn 23.0.0
[2025-10-12 21:30:00 +0000] [5835] [INFO] Listening at: http://0.0.0.0:5002 (5835)
[2025-10-12 21:30:00 +0000] [5835] [INFO] Using worker: sync
[2025-10-12 21:30:00 +0000] [5836] [INFO] Booting worker with pid: 5836
[2025-10-12 21:30:00 +0000] [5837] [INFO] Booting worker with pid: 5837
[2025-10-12 21:30:00 +0000] [5838] [INFO] Booting worker with pid: 5838
```

## 🔍 Quick Fix Commands

```bash
# 1. Navigate to correct directory
cd ~/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/LLMAcadamic-Advisor

# 2. Set Python path
export PYTHONPATH=.

# 3. Start with Gunicorn
gunicorn -w 3 -b 0.0.0.0:5002 api.api:app

# 4. Test the server
curl http://localhost:5002/llm/generate-response -X POST -H "Content-Type: application/json" -d '{"user_question":"test"}'
```

The key is being in the right directory and using the correct module path: `api.api:app`



