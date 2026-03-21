const express = require('express');
const mysql = require('mysql2');
const bcrypt = require('bcrypt');
const { exec } = require('child_process');
const saltRounds = 10;
const cors = require('cors');

const app = express();
app.use(express.json());
app.use(cors());

const dbConfig = {
    host: process.env.MYSQL_HOST || '127.0.0.1',
    user: process.env.MYSQL_USER || 'root',
    password: process.env.MYSQL_PASSWORD || '',
    database: process.env.MYSQL_DATABASE || 'Advising',
    port: process.env.MYSQL_PORT ? Number(process.env.MYSQL_PORT) : 3306
};

const disableDb = String(process.env.MYSQL_DISABLE || '').toLowerCase() === 'true';
const autoStartEnabled = String(process.env.MYSQL_AUTO_START || '').toLowerCase() !== 'false';
const isMac = process.platform === 'darwin';
let autoStartAttempted = false;
let connection = null;
let dbAvailable = false;

if (disableDb) {
    console.warn('⚠️ MYSQL_DISABLE=true → running without database. Auth endpoints will return 503.');
} else {
    const tryAutoStartMysql = (onDone) => {
        if (!isMac || !autoStartEnabled || autoStartAttempted) {
            return onDone(false);
        }
        autoStartAttempted = true;
        console.warn('⚠️ MySQL not running. Attempting to start via Homebrew...');
        exec('brew services start mysql', (err, stdout, stderr) => {
            if (err) {
                console.warn('⚠️ Failed to start MySQL via Homebrew.');
                if (stderr) {
                    console.warn(stderr.trim());
                }
                return onDone(false);
            }
            if (stdout) {
                console.log(stdout.trim());
            }
            setTimeout(() => onDone(true), 1500);
        });
    };

    const connectDb = () => {
        connection = mysql.createConnection(dbConfig);

connection.connect((err) => {
    if (err) {
                dbAvailable = false;
                const isConnRefused = err.code === 'ECONNREFUSED';
                if (isConnRefused) {
                    return tryAutoStartMysql((started) => {
                        if (started) {
                            return connectDb();
                        }
                        console.warn('⚠️ Database unavailable. Start MySQL or set MYSQL_DISABLE=true.');
                        console.warn(`   ${err.code || 'MYSQL_ERROR'} ${err.message || err.sqlMessage || ''}`);
                        if (err.address || err.port) {
                            console.warn(`   ${err.address || ''}:${err.port || ''}`);
                        }
                    });
                }
                console.warn('⚠️ Database unavailable. Start MySQL or set MYSQL_DISABLE=true.');
                console.warn(`   ${err.code || 'MYSQL_ERROR'} ${err.message || err.sqlMessage || ''}`);
                if (err.address || err.port) {
                    console.warn(`   ${err.address || ''}:${err.port || ''}`);
                }
        return;
    }
            dbAvailable = true;
    console.log('Connected to database');
});

        connection.on('error', (err) => {
            dbAvailable = false;
            console.error('Database connection error:', err);
        });
    };

    connectDb();
}

const ensureDbAvailable = (req, res, next) => {
    if (disableDb) {
        return res.status(503).json({ error: 'Database disabled. Set MYSQL_DISABLE=false to enable.' });
    }
    if (!dbAvailable || !connection) {
        return res.status(503).json({ error: 'Database unavailable. Start MySQL and retry.' });
    }
    return next();
};

app.post('/api/register', ensureDbAvailable, async (req, res) => {
    const { name, email, password, role, sex, dateOfBirth } = req.body;
    console.log("Role from client:", role);

    const allowedRoles = ['student', 'faculty', 'admin'];

    if (!allowedRoles.includes(role)) {
        return res.status(400).json({ error: 'Invalid role. Allowed roles are: student, faculty, admin' });
    }

    try {
        const hashedPassword = await bcrypt.hash(password, saltRounds);
        const query = 'INSERT INTO users (name, email, password, role, sex, dateOfBirth) VALUES (?, ?, ?, ?, ?, ?)';

        console.log("SQL Query:", query);
        console.log("Query Values:", [name, email, hashedPassword, role, sex, dateOfBirth]);

        connection.query(query, [name, email, hashedPassword, role, sex, dateOfBirth], (err, results) => {
            if (err) {
                console.error('Error inserting user: ', err.sqlMessage);
                return res.status(500).json({ error: 'Registration failed' });
            }
            console.log('User registered successfully');
            res.json({ message: 'User registered successfully' });
        });
    } catch (error) {
        console.error('Error hashing password:', error);
        res.status(500).json({ error: 'Registration is failed' });
    }
});

app.post('/api/login', ensureDbAvailable, async (req, res) => {
    const { email, password } = req.body;

    const query = 'SELECT * FROM users WHERE email = ?';

    connection.query(query, [email], async (err, results) => {
        if (err) {
            console.error('Error querying user: ', err);
            return res.status(500).json({ error: 'Login failed' });
        }

        if (results.length === 0) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        const user = results[0];
        
        // Try bcrypt first (server uses bcrypt)
        let passwordMatch = await bcrypt.compare(password, user.password);
        
        // If bcrypt fails, try bcryptjs (create_user.js uses bcryptjs)
        if (!passwordMatch) {
            try {
                const bcryptjs = require('bcryptjs');
                passwordMatch = await bcryptjs.compare(password, user.password);
            } catch (e) {
                // bcryptjs not available, continue with bcrypt result
            }
        }

        if (passwordMatch) {
            res.json({ message: 'Login successful', token: 'dummy-token-' + user.id, user: { id: user.id, name: user.name, email: user.email, role: user.role, sex: user.sex, dateOfBirth: user.dateOfBirth } }); // Send user data
        } else {
            res.status(401).json({ error: 'Invalid credentials' });
        }
    });
});

app.listen(5001, () => {
    console.log('Server started on port 5001');
});