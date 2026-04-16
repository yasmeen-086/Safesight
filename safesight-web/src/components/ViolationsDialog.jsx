import { useEffect } from 'react';
import './ViolationsDialog.css';

const MOCK_VIOLATIONS = [
    { time: '00:01:23', type: 'No Helmet', severity: 'high', zone: 'Zone A', worker: 'W-003', confidence: 94.2 },
    { time: '00:02:07', type: 'Restricted Area Entry', severity: 'high', zone: 'Zone C', worker: 'W-011', confidence: 91.8 },
    { time: '00:03:45', type: 'No Safety Vest', severity: 'medium', zone: 'Zone A', worker: 'W-007', confidence: 88.5 },
    { time: '00:05:12', type: 'Too Close to Machinery', severity: 'high', zone: 'Zone B', worker: 'W-003', confidence: 96.1 },
    { time: '00:06:34', type: 'No Helmet', severity: 'high', zone: 'Zone D', worker: 'W-019', confidence: 89.7 },
    { time: '00:08:55', type: 'No Safety Vest', severity: 'medium', zone: 'Zone B', worker: 'W-015', confidence: 85.3 },
    { time: '00:11:02', type: 'Idle in Hazard Zone', severity: 'low', zone: 'Zone C', worker: 'W-022', confidence: 78.9 },
];

export default function ViolationsDialog({ filename, onClose }) {
    const highCount = MOCK_VIOLATIONS.filter((v) => v.severity === 'high').length;
    const medCount = MOCK_VIOLATIONS.filter((v) => v.severity === 'medium').length;
    const lowCount = MOCK_VIOLATIONS.filter((v) => v.severity === 'low').length;

    useEffect(() => {
        const onKey = (e) => e.key === 'Escape' && onClose();
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [onClose]);

    return (
        <div className="dialog-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
            <div className="dialog" role="dialog" aria-modal="true">
                {/* Header */}
                <div className="dialog-header">
                    <div className="dialog-header-info">
                        <h2>Violation Report</h2>
                        <div className="dialog-meta">
                            <span>📄 {filename}</span>
                            <span>⏱ Processing time: 4.2s</span>
                            <span>🔍 {MOCK_VIOLATIONS.length} violations found</span>
                        </div>
                    </div>
                    <button className="dialog-close" onClick={onClose} aria-label="Close">
                        ✕
                    </button>
                </div>

                {/* Summary Stats */}
                <div className="dialog-stats">
                    <span className="stat-chip high">
                        <span className="stat-dot" />
                        {highCount} High
                    </span>
                    <span className="stat-chip medium">
                        <span className="stat-dot" />
                        {medCount} Medium
                    </span>
                    <span className="stat-chip low">
                        <span className="stat-dot" />
                        {lowCount} Low
                    </span>
                </div>

                {/* Table */}
                <div className="dialog-body">
                    <table className="violations-table">
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Violation</th>
                                <th>Severity</th>
                                <th>Zone</th>
                                <th>Worker</th>
                                <th>Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
                            {MOCK_VIOLATIONS.map((v, i) => (
                                <tr key={i}>
                                    <td>
                                        <span style={{ fontFamily: 'var(--font-heading)', fontSize: '0.78rem' }}>
                                            {v.time}
                                        </span>
                                    </td>
                                    <td>{v.type}</td>
                                    <td>
                                        <span className={`severity-badge ${v.severity}`}>
                                            {v.severity === 'high' ? '●' : v.severity === 'medium' ? '●' : '●'}
                                            {' '}{v.severity.charAt(0).toUpperCase() + v.severity.slice(1)}
                                        </span>
                                    </td>
                                    <td>{v.zone}</td>
                                    <td>
                                        <span style={{ fontFamily: 'var(--font-heading)', fontSize: '0.78rem' }}>
                                            {v.worker}
                                        </span>
                                    </td>
                                    <td>
                                        <span className="confidence">{v.confidence}%</span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Footer */}
                <div className="dialog-footer">
                    <button className="btn btn-outline" onClick={onClose}>
                        Close
                    </button>
                    <button className="btn btn-primary">
                        Export Report
                    </button>
                </div>
            </div>
        </div>
    );
}
