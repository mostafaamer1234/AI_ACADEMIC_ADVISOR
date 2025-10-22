# 🚀 AWS Deployment Guide - AI Academic Advisor

Complete step-by-step guide to deploy your application on AWS EC2.

## 📋 Prerequisites

- AWS Account
- EC2 instance (Ubuntu 22.04 LTS recommended)
- Domain name (optional, but recommended)
- SSH key pair for EC2 access

---

## 🖥️ Step 1: Launch EC2 Instance

### 1.1 Launch Instance
```
- AMI: Ubuntu Server 22.04 LTS (Free tier eligible)
- Instance Type: t3.micro (Free tier) or t2.medium+ for production
- Storage: 8 GB (Free tier) or 20 GB minimum for production
- Security Group: Configure ports (see below)
```

### 1.2 Security Group Configuration
Open these ports:
```
- SSH: 22 (Anywhere 0.0.0.0/0 for development, restrict for production)
- HTTP: 80 (0.0.0.0/0)
- HTTPS: 443 (0.0.0.0/0)
- Custom TCP: 5001 (for Node.js backend)
- Custom TCP: 5002 (for Flask LLM)
```

**Note:** For production, restrict SSH to your specific IP address.

---

## 🔧 Step 2: Connect and Initial Setup

### 2.1 Connect to EC2
```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### 2.2 Update System
```bash
sudo apt update
sudo apt upgrade -y
```

### 2.3 Install Required Software
```bash
# Install Node.js 18.x
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Python 3 and pip
sudo apt install -y python3 python3-pip python3-venv

# Install MySQL
sudo apt install -y mysql-server

# Install Nginx (reverse proxy)
sudo apt install -y nginx

# Install Git
sudo apt install -y git

# Install build tools (needed for some Python packages)
sudo apt install -y build-essential python3-dev libmysqlclient-dev

# Install graphviz (needed for networkx)
sudo apt install -y graphviz graphviz-dev
```

### 2.4 Memory Optimization for t3.micro
```bash
# Create swap file (important for t3.micro with limited RAM)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap is active
free -h
```

---

## 📥 Step 3: Clone Your Repository

```bash
# Create application directory
mkdir -p ~/flaskApp
cd ~/flaskApp

# Clone repository
git clone https://github.com/mostafaamer1234/AI_ACADEMIC_ADVISOR.git
cd AI_ACADEMIC_ADVISOR
```

---

## 🐍 Step 4: Setup Python Environment

### 4.1 Create Virtual Environment
```bash
cd ~/flaskApp/AI_ACADEMIC_ADVISOR
python3 -m venv venv
source venv/bin/activate
```

### 4.2 Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn  # For production server
```

### 4.3 Verify Installation
```bash
cd IAAS/LLMAcadamic-Advisor
python3 -c "from api.api import app; print('✅ Flask import successful')"
```

---

## 🗄️ Step 5: Setup MySQL Database

### 5.1 Secure MySQL Installation
```bash
sudo mysql_secure_installation
# Answer yes to all questions
```

### 5.2 Create Database and User
```bash
sudo mysql -u root -p
```

Then in MySQL:
```sql
CREATE DATABASE Advising;

CREATE USER 'advisor_user'@'localhost' IDENTIFIED BY 'Test1234!@#$';
GRANT ALL PRIVILEGES ON Advising.* TO 'advisor_user'@'localhost';
FLUSH PRIVILEGES;

USE Advising;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role ENUM('student', 'faculty', 'admin') NOT NULL,
    sex VARCHAR(10),
    dateOfBirth DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

EXIT;
```

### 5.3 Update Database Connection
Edit `IAAS/client/server/server.js`:
```javascript
const connection = mysql.createConnection({
    host: 'localhost',
    user: 'advisor_user',
    password: 'Test1234!@#$',
    database: 'Advising'
});
```

---

## 📦 Step 6: Setup Node.js Backend

### 6.1 Install Dependencies
```bash
cd ~/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/client
npm install

cd server
npm install
```

### 6.2 Test Backend
```bash
node server.js
# Should see: "Server started on port 5001" and "Connected to database"
# Press Ctrl+C to stop
```

---

## ⚛️ Step 7: Build React Frontend

### 7.1 Build Production Version (Recommended: Build Locally)
**For t3.micro instances, build locally to avoid memory issues:**

**On your local machine:**
```bash
cd /path/to/your/project/IAAS/client
npm run build
```

**Upload build folder to EC2:**
```bash
scp -i ~/Downloads/MYFirst.pem -r build ubuntu@YOUR-EC2-IP:~/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/client/
```

**Alternative: Build on EC2 (if you have sufficient resources):**
```bash
cd ~/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/client
npm run build
```

This creates optimized static files in `build/` directory.

---

## 🎯 Step 8: Setup Process Managers

We'll use **systemd** to manage all three services.

### 8.1 Flask LLM Service
Create `/etc/systemd/system/flask-llm.service`:
```bash
sudo nano /etc/systemd/system/flask-llm.service
```

Content:
```ini
[Unit]
Description=Flask LLM Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/LLMAcadamic-Advisor
Environment="PATH=/home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/venv/bin"
Environment="PYTHONPATH=/home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/LLMAcadamic-Advisor"
ExecStart=/home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/venv/bin/gunicorn -w 1 -b 127.0.0.1:5002 api.api:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 8.2 Node.js Backend Service
Create `/etc/systemd/system/nodejs-backend.service`:
```bash
sudo nano /etc/systemd/system/nodejs-backend.service
```

Content:
```ini
[Unit]
Description=Node.js Backend Server
After=network.target mysql.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/client/server
ExecStart=/usr/bin/node server.js
Restart=always
RestartSec=10
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
```

### 8.3 Enable and Start Services
```bash
sudo systemctl daemon-reload

# Flask LLM Server
sudo systemctl enable flask-llm
sudo systemctl start flask-llm
sudo systemctl status flask-llm

# Node.js Backend
sudo systemctl enable nodejs-backend
sudo systemctl start nodejs-backend
sudo systemctl status nodejs-backend
```

---

## 🌐 Step 9: Setup Nginx Reverse Proxy

### 9.1 Create Nginx Configuration
```bash
sudo nano /etc/nginx/sites-available/academic-advisor
```

Content:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Or use EC2 public IP

    # Serve React Frontend
    location / {
        root /home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/client/build;
        try_files $uri /index.html;
    }

    # Proxy to Node.js Backend (Login/Auth)
    location /api/ {
        proxy_pass http://localhost:5001/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Proxy to Flask LLM Server
    location /llm/ {
        proxy_pass http://localhost:5002/llm/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

### 9.2 Enable Configuration
```bash
sudo ln -s /etc/nginx/sites-available/academic-advisor /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

---

## 🔒 Step 10: SSL Certificate (HTTPS)

### 10.1 Install Certbot
```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 10.2 Obtain Certificate
```bash
sudo certbot --nginx -d your-domain.com
# Follow prompts
```

---

## ✅ Step 11: Verify Deployment

### 11.1 Check All Services
```bash
# Check Flask LLM
sudo systemctl status flask-llm
curl http://localhost:5002/llm/generate-response -X POST -H "Content-Type: application/json" -d '{"user_question":"test"}'

# Check Node.js Backend
sudo systemctl status nodejs-backend
curl http://localhost:5001/api/login -X POST -H "Content-Type: application/json" -d '{"email":"test@test.com","password":"test"}'

# Check Nginx
sudo systemctl status nginx
```

### 11.2 Access Application
```
http://your-ec2-public-ip
# Or
https://your-domain.com
```

---

## 🔧 Troubleshooting

### View Service Logs
```bash
# Flask LLM logs
sudo journalctl -u flask-llm -f

# Node.js Backend logs
sudo journalctl -u nodejs-backend -f

# Nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```

### Restart Services
```bash
sudo systemctl restart flask-llm
sudo systemctl restart nodejs-backend
sudo systemctl restart nginx
```

### Check Port Usage
```bash
sudo lsof -i :5001
sudo lsof -i :5002
sudo lsof -i :80
```

---

## 📊 Step 12: Monitoring (Optional)

### Install PM2 (Alternative to systemd)
```bash
sudo npm install -g pm2

# Start services
pm2 start /home/ubuntu/flaskApp/AI_ACADEMIC_ADVISOR/IAAS/client/server/server.js --name "nodejs-backend"
pm2 save
pm2 startup
```

---

## 🔄 Step 13: Updates and Maintenance

### Update Application
```bash
cd ~/flaskApp/AI_ACADEMIC_ADVISOR
git pull origin main

# Rebuild frontend
cd IAAS/client
npm run build

# Restart services
sudo systemctl restart flask-llm
sudo systemctl restart nodejs-backend
```

---

## 📋 Deployment Checklist

- [ ] EC2 instance launched with correct security groups (t3.micro for free tier)
- [ ] All software installed (Node.js, Python, MySQL, Nginx)
- [ ] Swap file created for t3.micro memory optimization
- [ ] Repository cloned
- [ ] Python virtual environment created
- [ ] Python dependencies installed
- [ ] MySQL database created and configured
- [ ] Node.js dependencies installed
- [ ] React app built (locally recommended for t3.micro)
- [ ] Build folder uploaded to EC2 via SCP
- [ ] Systemd services created and running
- [ ] Nginx configured and running
- [ ] SSL certificate installed (optional)
- [ ] Application accessible via browser
- [ ] All three services communicating correctly

---

## 🆘 Common Issues

### Issue: Services won't start
**Solution:** Check logs with `journalctl -u service-name -f`

### Issue: Database connection failed
**Solution:** Verify MySQL credentials in `server.js`

### Issue: Port already in use
**Solution:** `sudo lsof -i :PORT` and kill process

### Issue: Frontend can't connect to backend
**Solution:** Check nginx configuration and proxy settings

### Issue: t3.micro out of memory during build
**Solution:** Build React app locally and upload via SCP

### Issue: MySQL authentication error
**Solution:** Run `ALTER USER 'advisor_user'@'localhost' IDENTIFIED WITH mysql_native_password BY 'Test1234!@#$';`

### Issue: SSH connection timeout
**Solution:** Use public IP address, not private IP. Check security group allows SSH from your IP.

### Issue: Permission denied for .pem file
**Solution:** Run `chmod 400 ~/Downloads/MYFirst.pem`

---

## 🎉 Success!

Your application should now be running at:
- **Frontend**: http://your-ec2-ip (or your-domain.com)
- **Backend API**: Proxied through nginx at `/api/`
- **LLM API**: Proxied through nginx at `/llm/`

All services are managed by systemd and will auto-restart on failure or reboot!
