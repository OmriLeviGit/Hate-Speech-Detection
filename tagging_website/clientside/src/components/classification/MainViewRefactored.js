import './MainViewRefactored.css';
import { useEffect, useState } from "react";
import { toast, ToastContainer } from "react-toastify";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { fetchTweet, submitClassification, fetchFeatures, fetchClassificationCount } from "../../services/api";
import FeatureButton from './FeatureButton';
import Panel from "./Panels/Panel";
import "react-toastify/dist/ReactToastify.css";

// ToDo - Load the feature params from the server
// ToDo - Enable selecting features only when tagging as positive
// ToDo - Show classifications count to the user
// ToDo - Add a sign out button
// ToDo - Add profile button
// ToDo - Show the personal statistics
// ToDo - right the todos for Pro users

const MainViewRefactored = ({ token, setToken, setPasscode, isPro }) => {

    // Helps the toggle button of tagging as antisemitic or not antisemitic
    const [isAntisemitic, setIsAntisemitic] = useState(false);

    // Helps with showing/hiding the stats panel
    const [isPanelOpen, setIsPanelOpen] = useState(false);

    // Helps with copying the text of the current displayed tweet
    const [copyStatus, setCopyStatus] = useState(false);

    // Helps with showing how many tags were made by the current logged-in user
    const [classificationCount, setClassificationCount] = useState(0);

    // State for managing the tweet
    const [tweet, setTweet] = useState(null);

    // Disables the buttons while sending a tag submission
    const [loading, setLoading] = useState(false);

    // Disables the classifications button when there are no tweets to tag
    const [doneTagging, setDoneTagging] = useState(false);

    // Stores the feature list
    const [featuresList, setFeaturesList] = useState([]);

    // Stores the selected features of a tweet before submission, if is being tagged "Positive"
    const [selectedFeatures, setSelectedFeatures] = useState([]);


    const onCopyText = () => {
        setCopyStatus(true);
        setTimeout(() => setCopyStatus(false), 2000);
    };


    const handleFeatureSelection = (feature) => {
        setSelectedFeatures((prevFeatures) =>
            prevFeatures.includes(feature)
                ? prevFeatures.filter(f => f !== feature)  // Remove if already selected
                : [...prevFeatures, feature]  // Add if not selected
        );
    };


    // Sends a request to the server to fetch a tweet to tag
    const getNewTweet = async () => {
        setLoading(true);
        const resj = await fetchTweet(token);

        if (!resj) {
            setTweet({ content: "Error connecting to server!" });
        } else if (resj.error) {
            setDoneTagging(true);
            setTweet({ content: "No tweets left to classify! 🎉" });
        } else {
            setDoneTagging(false);
            setTweet({
                tweetId: resj.id,
                content: resj.content,
                tweetURL: resj.tweet_url,
            });
        }

        setLoading(false);
    };


    const fetchClassificationCountData = async () => {
        const resj = await fetchClassificationCount(token);
        if (resj) {
            setClassificationCount(resj.count);
        }
    };

    // Loads the feature list from the server
    const loadFeatures = async () => {
        const resj = await fetchFeatures();
        setFeaturesList(resj.map(feature => feature[1]));
    };


    // Sends a request to the server to tag the current tweet
    const submitTweetTagging = async (classification, features) => {
        if (!tweet) return;

        setLoading(true);
        const success = await submitClassification(token, tweet.tweetId, classification, features);
        if (success) {
            // Shows the user a success message on screens
            toast.success("Classification submitted!", { autoClose: 2000 });
            // Resets feature selection
            setSelectedFeatures([]);
            // Fetches next tweet
            await getNewTweet();
        } else {
            toast.error("Error submitting classification.");
        }
        setLoading(false);
        setIsAntisemitic(false);
    };


    // Handles sign out
    const signOut = () => {
        setPasscode(null);
        setToken(null);
    }


    // Happens on component mount
    useEffect(() => {
        getNewTweet();
        loadFeatures();
        fetchClassificationCountData();
    }, []);


    return (
        <div id="form-frame">

            <h1 className="form-title">{isPro ? "Tweet Classifier - Pro" : " Tweet Classifier"}</h1>

            {loading ? <p>Loading tweet...</p> :
                <p>{tweet ? (
                    <p>{tweet.content}</p>
                ) : (
                    <p style={{ color: "gray", fontStyle: "italic" }}>All tweets are classified! 🎉</p>
                )}
                </p>}


            {/* ToDo - Add the copy button (copies tweet's content)*/}
            <div className="copy-to-clip-div">
                <CopyToClipboard text={tweet ? tweet.content : ""} onCopy={() => {
                    toast.success("Text copied to clipboard", { autoClose: 2000 });
                }}>
                    <button className="copy-to-clip-btn" type="button">
                        <span className="bi bi-clipboard"></span>
                        <span style={{ paddingLeft: "5%" }}></span>
                        Copy
                    </button>
                </CopyToClipboard>
            </div>


            {/* Toggle button for tagging as Antisemitic or Not Antisemitic*/}
            <div className="form-check form-switch form-switch-md">

                <input
                    className="form-check-input"
                    type="checkbox"
                    id="flexSwitchCheckDefault"
                    checked={isAntisemitic}
                    onChange={() => setIsAntisemitic(!isAntisemitic)}
                    disabled={doneTagging}
                />
                <label className="form-check-label" htmlFor="flexSwitchCheckDefault">
                    {isAntisemitic ? "Antisemitic" : "Not Antisemitic"}
                </label>

            </div>


            <div className="classification-zone">

                <button
                    type="submit" id="classify-btn"
                    className="submit-button classify-submit-button"
                    onClick={() =>
                        submitTweetTagging(isAntisemitic ? "Positive" : "Negative", selectedFeatures)}
                    disabled={loading || doneTagging || (isAntisemitic && selectedFeatures.length === 0)}
                >
                    Classify As {isAntisemitic ? "" : "Not "}Antisemitic
                </button>

                <button
                    id="not-sure-btn"
                    className="small-side-button"
                    type="button"
                    disabled={loading || doneTagging || isAntisemitic}
                    onClick={() => submitTweetTagging("Uncertain", [])}
                >
                    <span className="bi bi-question-octagon-fill"/>
                    <span style={{paddingLeft: "3%"}}/>
                    <span>Uncertain</span>
                </button>

                <span style={{paddingLeft: "2%"}}/>

                <button
                    id="irrelevant-btn"
                    className="small-side-button"
                    type="button"
                    disabled={loading || doneTagging || isAntisemitic}
                    onClick={() => submitTweetTagging("Irrelevant", [])}
                >
                    <span>Irrelevant</span>
                    <span style={{paddingLeft: "3%"}}/>
                    <span className="bi bi-trash-fill"/>
                </button>

            </div>


            {isAntisemitic && (
                <div className="features-container">
                    <div className="select-feature">
                        Select at least one feature before classifying as antisemitic
                    </div>
                    {featuresList.map((feature) => (
                        <FeatureButton
                            key={feature}
                            feature={{ description: feature, checked: selectedFeatures.includes(feature) }}
                            disabled={loading}
                            setFeature={(feature, bool) => handleFeatureSelection(feature.description)}
                        />
                    ))}
                </div>
            )}


            {/* ToDo - Consider adding "x classifications made out of y classifications needed" */}
            <div>
                Classifications Made: {classificationCount}
            </div>

            <div className="bottom-container">

                {/* Sign out button */}
                <button id="sign-out-btn"
                        className="bottom-container-button"
                        type="button"
                        onClick={signOut}>
                    <span className="bi bi-door-closed" />
                    <span style={{ paddingLeft: "3%" }} />
                    <span>Sign Out</span>
                </button>

                <button id="panel-btn"
                        className="bottom-container-button"
                        type="button"
                        onClick={() => setIsPanelOpen(true)}>
                    <span>{isPro ? "Admin Panel" : "Profile"}</span>
                    <span style={{ paddingLeft: "5%" }} />
                    <span className="bi bi-person-circle" />
                </button>

            </div>

            {isPanelOpen && (
                <Panel token={token} isPro={isPro} onClose={() => setIsPanelOpen(false)} />
            )}

            <ToastContainer/>
        </div>
    );
};

export default MainViewRefactored;
