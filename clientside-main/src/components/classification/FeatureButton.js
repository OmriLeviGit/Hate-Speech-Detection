import "./FeatureButton.css";

const FeatureButton = ({ feature, disabled, setFeature }) => {
    const clicked = (bool) => {
        if (disabled) return;
        // Set the feature to the new value
        setFeature(feature, bool);
    };


    const getClassName = () => {
        let newClsName = "bi bi-";
        newClsName += feature.checked ? "check" : "x";
        newClsName += "-circle-fill feature-mark ";
        if (feature.checked) {
            newClsName += `feature-mark-checked${disabled ? "-disabled" : ""}`;
        }
        else {
            newClsName += `feature-mark-unchecked${disabled ? "-disabled" : ""}`;
        }
        return newClsName;
    }

    return (
        <div className={`feature-row ${disabled ? 'feature-row-disabled' : ""}`} onClick={() => clicked(!feature.checked)}>
            <button
                type="button"
                disabled={disabled}
                className="feature-button"
            >
                {feature.description}
            </button>
            <div className="button-container">
                <i
                    className={getClassName()}
                ></i>
            </div>
        </div>
    );
};

export default FeatureButton;