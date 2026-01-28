import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard'; // AI Advisor page (student dashboard)
import FacultyDashboard from './pages/FacultyDashboard'; // Faculty Advisor page
import AdminDashboard from './pages/AdminDashboard'; // Admin Dashboard
import Profile from './pages/Profile'; // Profile page

const AppRouter = () => {
  return (
    <Router>
      <Routes>
        {/* AI Advisor Route (default) */}
        <Route path="/" element={<Dashboard />} />
        <Route path="/dashboard" element={<Dashboard />} />

        {/* Faculty Advisor Route */}
        <Route path="/faculty-dashboard" element={<FacultyDashboard />} />

        {/* Admin Dashboard Route */}
        <Route path="/admin-dashboard" element={<AdminDashboard />} />

        {/* Profile Route */}
        <Route path="/profile" element={<Profile />} />

        {/* Catch-All Route */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
};

export default AppRouter;