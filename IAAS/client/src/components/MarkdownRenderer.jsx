import React from 'react';
import '../styles/MarkdownRenderer.css';

const MarkdownRenderer = ({ content }) => {
    const formatContent = (text) => {
        if (!text) return null;

        // Remove markdown headers (##, ###, etc.)
        text = text.replace(/^#{1,6}\s+/gm, '');
        
        // Split by double newlines to create paragraphs
        const sections = text.split(/\n\n+/);
        
        return sections.map((section, sectionIndex) => {
            const lines = section.split('\n');
            const formattedLines = [];
            
            lines.forEach((line, lineIndex) => {
                const trimmedLine = line.trim();
                if (!trimmedLine) return;
                
                // Check for bold text (remove ** and make it a header style)
                if (trimmedLine.match(/^\*\*[^*]+\*\*$/)) {
                    const headerText = trimmedLine.replace(/\*\*/g, '').trim();
                    formattedLines.push(
                        <h3 key={`${sectionIndex}-${lineIndex}`} className="simple-header">
                            {headerText}
                        </h3>
                    );
                }
                // Check for lines that start with ** (bold headers) followed by text
                else if (trimmedLine.match(/^\*\*[^*]+\*\*\s*/)) {
                    const headerText = trimmedLine.replace(/\*\*/g, '').trim();
                    formattedLines.push(
                        <h3 key={`${sectionIndex}-${lineIndex}`} className="simple-header">
                            {formatInlineMarkdown(headerText)}
                        </h3>
                    );
                }
                // Check for list items starting with - or *
                else if (trimmedLine.match(/^[-*]\s+/)) {
                    const listItem = trimmedLine.replace(/^[-*]\s+/, '').trim();
                    formattedLines.push(
                        <div key={`${sectionIndex}-${lineIndex}`} className="simple-list-item">
                            <span className="list-content">{formatInlineMarkdown(listItem)}</span>
                        </div>
                    );
                }
                // Check for numbered list items
                else if (trimmedLine.match(/^\d+\.\s+/)) {
                    const listItem = trimmedLine.replace(/^\d+\.\s+/, '').trim();
                    formattedLines.push(
                        <div key={`${sectionIndex}-${lineIndex}`} className="simple-list-item">
                            <span className="list-content">{formatInlineMarkdown(listItem)}</span>
                        </div>
                    );
                }
                // Regular paragraph
                else {
                    const cleanedLine = trimmedLine.replace(/^\*\*|\*\*$/g, ''); // Remove remaining **
                    formattedLines.push(
                        <p key={`${sectionIndex}-${lineIndex}`} className="simple-paragraph">
                            {formatInlineMarkdown(cleanedLine)}
                        </p>
                    );
                }
            });
            
            return (
                <div key={sectionIndex} className="simple-section">
                    {formattedLines}
                </div>
            );
        });
    };

    const formatInlineMarkdown = (text) => {
        if (!text) return '';
        
        // Remove all ** markdown syntax
        text = text.replace(/\*\*/g, '');
        
        // Format course codes
        return formatCourseCodes(text);
    };

    const formatCourseCodes = (text) => {
        // More comprehensive regex to match course codes with descriptions, units, and prerequisites
        // Pattern: CMPSC 221: Course Name (X.XX units, prerequisites: ...)
        const courseRegex = /([A-Z]{2,}\s+\d{3}[A-Z]*):\s*([^(]+?)\s*(\([^)]+\))/g;
        const parts = [];
        let lastIndex = 0;
        let match;
        
        while ((match = courseRegex.exec(text)) !== null) {
            // Add text before course code
            if (match.index > lastIndex) {
                const beforeText = text.substring(lastIndex, match.index);
                // Also handle standalone course codes without description
                parts.push(formatStandaloneCourseCodes(beforeText));
            }
            
            // Add formatted course code with description
            const courseCode = match[1];
            const courseName = match[2].trim();
            const details = match[3];
            
            parts.push(
                <span key={`course-${match.index}`} className="course-listing">
                    <span className="course-code">{courseCode}</span>
                    <span className="course-name">{courseName}</span>
                    <span className="course-details">{details}</span>
                </span>
            );
            
            lastIndex = courseRegex.lastIndex;
        }
        
        // Add remaining text
        if (lastIndex < text.length) {
            const remaining = text.substring(lastIndex);
            parts.push(formatStandaloneCourseCodes(remaining));
        }
        
        return parts.length > 0 ? parts : text;
    };

    const formatStandaloneCourseCodes = (text) => {
        if (!text || typeof text !== 'string') return text;
        
        // Format standalone course codes like "CMPSC 221" without descriptions
        const standaloneRegex = /\b([A-Z]{2,}\s+\d{3}[A-Z]*)\b/g;
        const parts = [];
        let lastIndex = 0;
        let match;
        
        while ((match = standaloneRegex.exec(text)) !== null) {
            // Add text before course code
            if (match.index > lastIndex) {
                parts.push(text.substring(lastIndex, match.index));
            }
            // Add formatted course code
            parts.push(
                <span key={`standalone-${match.index}`} className="course-code-inline">
                    {match[1]}
                </span>
            );
            lastIndex = standaloneRegex.lastIndex;
        }
        
        // Add remaining text
        if (lastIndex < text.length) {
            parts.push(text.substring(lastIndex));
        }
        
        return parts.length > 0 ? parts : text;
    };

    return <div className="markdown-content">{formatContent(content)}</div>;
};

export default MarkdownRenderer;

