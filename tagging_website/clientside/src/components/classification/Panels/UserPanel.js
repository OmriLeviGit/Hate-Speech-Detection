import { useEffect, useState } from 'react';
import './UserPanel.css';

const UserPanel = ({ token }) => {
    const [user, setUser] = useState({
        personalClassifications: 0,
        positiveClassified: 0,
        negativeClassified: 0,
        timeLeft: 0,
        tweetsRemaining: 0,
        averageTime: 0,
        irrelevantClassified: 0,
    });


    const getUserStats = async () => {
        // Get new tweet from server and set tweet state
        fetch(window.API_URL + '/get_user_panel', {
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
                    // Create tweet object
                    const currUser = {
                        personalClassifications: resj.total,
                        positiveClassified: resj.pos,
                        negativeClassified: resj.neg,
                        timeLeft: resj.time,
                        tweetsRemaining: resj.remain,
                        averageTime: resj.avg,
                        irrelevantClassified: resj.irr,
                    };
                    setUser(currUser);
                });
            }
        });
    }

    useEffect(() => {
        getUserStats();
    }, [])

    return (
        <div className="user-panel-container">
            <h2>My Stats</h2>
            <table className="user-panel-table">
                <thead>
                <tr>
                    <th>Classifications Left</th>
                    <th>Days Left</th>
                    <th>No. Classified</th>
                    <th>No. Positive</th>
                    <th>No. Negative</th>
                    <th>No. Irrelevant</th>
                    <th>Average Time (seconds)</th>
                    <th>% Positive</th>
                    <th>% Negative</th>
                    <th>% Irrelevant</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                    <td>{user.tweetsRemaining}</td>
                    <td>{user.timeLeft}</td>
                    <td>{user.personalClassifications}</td>
                    <td>{user.positiveClassified}</td>
                    <td>{user.negativeClassified}</td>
                    <td>{user.irrelevantClassified}</td>
                    <td>{user.averageTime > 0 ? user.averageTime : 0}</td>
                    <td>{user.personalClassifications > 0 ? ((user.positiveClassified / user.personalClassifications) * 100).toFixed(2) : 0}%</td>
                    <td>{user.personalClassifications > 0 ? ((user.negativeClassified / user.personalClassifications) * 100).toFixed(2) : 0}%</td>
                    <td>{user.personalClassifications > 0 ? ((user.irrelevantClassified / user.personalClassifications) * 100).toFixed(2) : 0}%</td>
                </tr>
                </tbody>
            </table>
        </div>
    );
};

export default UserPanel;
