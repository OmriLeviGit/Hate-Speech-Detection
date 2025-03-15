import "./Panel.css";
import ProPanel from "./ProPanel"
import UserPanel from "./UserPanel"

const Panel = ({ onClose, token, showAdminPanel }) => {
    return (
        <div className="main-panel">
            <div className="modal-content">
                <span className="close" onClick={onClose}>&times;</span>
                {showAdminPanel ? <ProPanel token={token} /> : <UserPanel token={token} />}
            </div>
        </div>
    );
};


export default Panel;
