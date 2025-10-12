# 🔧 Troubleshooting Guide

## Quick Fixes for Common Issues

### 🚨 **"Cannot log in" Error**

**Symptoms:**
- Login page shows "Invalid credentials" or doesn't respond
- Console shows network errors to `localhost:5001`

**Solution:**
```bash
# Check if backend server is running
lsof -ti:5001

# If not running, start it manually:
npm run start:backend

# Or restart everything:
npm start
```

---

### 🚨 **"Error occurred while generating graduation plan" Error**

**Symptoms:**
- PDF upload fails
- Console shows CORS errors or "Failed to fetch"
- Error messages about port 5000 or 5002

**Solutions:**

#### 1. **CORS Error (Most Common)**
```
Access to fetch at 'http://127.0.0.1:5002/llm/setup-environment' from origin 'http://localhost:3000' has been blocked by CORS policy
```

**Fix:** The LLM server isn't running or is on wrong port
```bash
# Check if LLM server is running on port 5002
lsof -ti:5002

# If not running, restart everything:
npm start
```

#### 2. **Port 5000 Conflict (macOS)**
```
POST http://127.0.0.1:5000/llm/setup-environment net::ERR_FAILED 403 (Forbidden)
```

**Fix:** macOS Control Center uses port 5000. Our app uses port 5002.
```bash
# Kill anything on port 5000
lsof -ti:5000 | xargs kill 2>/dev/null

# Restart the application
npm start
```

#### 3. **Python Command Not Found**
```
pyenv: python: command not found
```

**Fix:** The script uses `python3` instead of `python`
```bash
# Check if python3 is available
which python3

# If not available, install Python3 or configure pyenv
```

---

### 🚨 **"ModuleNotFoundError: No module named 'LLM'" Error**

**Symptoms:**
- Flask server fails to start
- Python import errors

**Fix:** The Flask server needs to run from the correct directory
```bash
# Make sure you're running from the client directory
cd /Users/mostafa/newFianlLLMRepo/IAAS/client

# Use npm start (not manual python commands)
npm start
```

---

### 🚨 **"Connected to database" but login still fails**

**Symptoms:**
- Backend shows "Connected to database"
- But login returns "Invalid credentials"

**Fix:** Check if you have user accounts in the database
```bash
# Connect to MySQL and check users table
/Applications/XAMPP/xamppfiles/bin/mysql -u root -e "USE Advising; SELECT email, role FROM users;"
```

**If no users exist, create a test account:**
```sql
-- Connect to MySQL and run:
USE Advising;
INSERT INTO users (name, email, password, role, sex, dateOfBirth) 
VALUES ('Test User', 'test@test.com', '$2b$10$example_hash_here', 'student', 'M', '1990-01-01');
```

---

## 🔍 **Diagnostic Commands**

### Check All Servers
```bash
# Check if all three servers are running
echo "Frontend (3000):" && lsof -ti:3000
echo "Backend (5001):" && lsof -ti:5001  
echo "LLM Server (5002):" && lsof -ti:5002
```

### Check MySQL
```bash
# Check if MySQL is running
pgrep -f mysql

# Check if Advising database exists
/Applications/XAMPP/xamppfiles/bin/mysql -u root -e "SHOW DATABASES LIKE 'Advising';"
```

### Check Python Environment
```bash
# Check Python version
python3 --version

# Check if required modules can be imported
cd ../LLMAcadamic-Advisor && python3 -c "from LLM.main_copy import setup_environment; print('✅ Import successful')"
```

---

## 🚀 **Complete Reset Procedure**

If nothing else works, follow this complete reset:

```bash
# 1. Stop all servers
lsof -ti:3000,5001,5002 | xargs kill 2>/dev/null

# 2. Go to client directory
cd /Users/mostafa/newFianlLLMRepo/IAAS/client

# 3. Run setup script
./setup.sh

# 4. Start everything
npm start

# 5. Wait for all servers to start (should see all three colors)
# 6. Go to http://localhost:3000 and test
```

---

## 📋 **Prevention Checklist**

Before running the application, ensure:

- ✅ **MySQL/XAMPP is running**
- ✅ **You're in the correct directory** (`/Users/mostafa/newFianlLLMRepo/IAAS/client`)
- ✅ **All dependencies are installed** (`npm install`)
- ✅ **Python3 is available** (`python3 --version`)
- ✅ **Ports 3000, 5001, 5002 are available**
- ✅ **Required files exist** (CSV file, database)

---

## 🆘 **Still Having Issues?**

1. **Run the setup script first:** `./setup.sh`
2. **Check the browser console** for specific error messages
3. **Check the terminal output** for server errors
4. **Try the complete reset procedure** above
5. **Make sure you're using the latest code** (all fixes are already applied)

---

## 📞 **Common Error Messages & Solutions**

| Error Message | Solution |
|---------------|----------|
| `CORS policy: No 'Access-Control-Allow-Origin'` | LLM server not running on port 5002 |
| `POST net::ERR_FAILED 403` | Port conflict, restart with `npm start` |
| `pyenv: python: command not found` | Use `python3` (already fixed in package.json) |
| `ModuleNotFoundError: No module named 'LLM'` | Run from correct directory with `npm start` |
| `Invalid credentials` | Check database connection and user accounts |
| `Stream has ended unexpectedly` | Invalid PDF file, try a different PDF |

**Remember: Always use `npm start` from the `/IAAS/client` directory!**
