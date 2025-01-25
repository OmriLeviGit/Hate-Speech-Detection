import './ProPanel.css';
import { useEffect, useState } from 'react';

const ProPanel = ({ token }) => {
    const [users, setUsers] = useState([]);
    const [totalClassified, setTotalClassified] = useState(0);
    const [totalPositive, setTotalPositive] = useState(0);
    const [totalNegative, setTotalNegative] = useState(0);
    const [totalIrrelevant, setTotalIrrelevant] = useState(0);

    const getProStats = async () => {
        // Get new tweet from server and set tweet state
        fetch(window.API_URL + '/get_pro_panel', {
            method: "GET", headers: {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token,
            }
        }).then(response => {
            // If response is ok, get new tweet and set tweet state
            if (response.ok) {
                response.json().then(resj => {
                    if (resj.error) {
                        return;
                    }
                    setUsers(resj.users);
                    setTotalClassified(resj.total);
                    setTotalPositive(resj.total_pos);
                    setTotalNegative(resj.total_neg);
                    setTotalIrrelevant(resj.total_irr);
                });
            }
        });
    }

    useEffect(() => {
        getProStats();
    }, [])

    let totalTime = 0;
    let overallTotalClassified = 0;

    users.forEach(user => {
        // Convert average time to milliseconds and sum
        totalTime += parseFloat(user.averageTime) * user.personalClassifications;
        overallTotalClassified += user.personalClassifications;
    });

    const totalUsers = users.length;
    const avgTime = overallTotalClassified > 0 ? (totalTime / overallTotalClassified).toFixed(2) : 0;
    const percentPositive = totalClassified > 0 ? (totalPositive / totalClassified * 100).toFixed(2) : 0;
    const percentNegative = totalClassified > 0 ? (totalNegative / totalClassified * 100).toFixed(2) : 0;
    const percentIrrelevant = totalClassified > 0 ? (totalIrrelevant / totalClassified * 100).toFixed(2) : 0;



    return (
        <div className="pro-panel-container">
            <h2>Overall Statistics for {totalUsers} Users</h2>
            <table className="pro-panel-table">
                <thead>
                    <tr>
                        <th>Total Classified</th>
                        <th>No. Positive</th>
                        <th>No. Negative</th>
                        <th>No. Irrelevant</th>
                        <th>% Positive</th>
                        <th>% Negative</th>
                        <th>% Irrelevant</th>
                        <th>Average Time</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{totalClassified}</td>
                        <td>{totalPositive}</td>
                        <td>{totalNegative}</td>
                        <td>{totalIrrelevant}</td>
                        <td>{percentPositive}%</td>
                        <td>{percentNegative}%</td>
                        <td>{percentIrrelevant}%</td>
                        <td>{avgTime > 0 ? avgTime : 0}</td>
                    </tr>
                </tbody>
            </table>

            <br></br>
            <h2>Users Stats</h2>
            <table className="pro-panel-table">
                <thead>
                    <tr>
                        <th>Email</th>
                        <th>No. Classified</th>
                        <th>No. Positive</th>
                        <th>No. Negative</th>
                        <th>No. Irrelevant</th>
                        <th>Average Time</th>
                        <th>% Positive</th>
                        <th>% Negative</th>
                        <th>% Irrelevant</th>
                    </tr>
                </thead>
                <tbody>
                    {users.map((user, index) => (
                        <tr key={index}>
                            <td>{user.email}</td>
                            <td>{user.personalClassifications}</td>
                            <td>{user.positiveClassified}</td>
                            <td>{user.negativeClassified}</td>
                            <td>{user.irrelevantClassified}</td>
                            <td>{user.averageTime > 0 ? user.averageTime : 0}</td>
                            <td>{user.personalClassifications > 0 ? ((user.positiveClassified / user.personalClassifications) * 100).toFixed(2) : 0}%</td>
                            <td>{user.personalClassifications > 0 ? ((user.negativeClassified / user.personalClassifications) * 100).toFixed(2) : 0}%</td>
                            <td>{user.personalClassifications > 0 ? ((user.irrelevantClassified / user.personalClassifications) * 100).toFixed(2) : 0}%</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default ProPanel;
