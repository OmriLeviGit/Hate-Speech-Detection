import "./Panel.css";
import ProPanel from "./ProPanel"
import UserPanel from "./UserPanel"

const Panel = ({ passcode, onClose, isPro, token }) => {


    return (
        <div className="main-panel">
            <div className="modal-content">
                <span className="close" onClick={onClose}>&times;</span>
                {isPro ? (
                    <ProPanel token={token} />
                ) : (
                    <UserPanel token={token} />
                )}
            </div>
        </div>
    );
};

export default Panel;
