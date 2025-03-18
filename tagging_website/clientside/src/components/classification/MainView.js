import { useEffect, useState } from "react";
import { toast, ToastContainer } from "react-toastify";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { fetchTweet, submitClassification, fetchFeatures, fetchClassificationCount} from "../../services/api";
import FeatureButton from './FeatureButton';
import Panel from "./Panels/Panel";
import "react-toastify/dist/ReactToastify.css";
import Tweet from "./Tweet";
import './MainView.css';

const MainView = ({ token, setToken, passcode, setPasscode, isPro }) => {
    // State management
    const [tweet, setTweet] = useState({
        tweetId: '',
        content: '',
        tweetURL: ''
    });
    const [isAntisemitic, setIsAntisemitic] = useState(false);
    const [selectedFeatures, setSelectedFeatures] = useState([]);
    const [featuresList, setFeaturesList] = useState([]);

    // UI state
    const [isUserPanelOpen, setIsUserPanelOpen] = useState(false);
    const [isAdminPanelOpen, setIsAdminPanelOpen] = useState(false);
    const [loading, setLoading] = useState(false);
    const [doneTagging, setDoneTagging] = useState(false);
    const [classificationCount, setClassificationCount] = useState(0);

    // Tracking state
    const [startTime, setStartTime] = useState(null);
    const [copyStatus, setCopyStatus] = useState(false);

    /* Helps with copying the text of the current displayed tweet */
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

    const fetchClassificationCountData = async () => {
        const resj = await fetchClassificationCount(token);
        if (resj) {
            setClassificationCount(resj.count);
        }
    };

    // Data fetching functions
    const getNewTweet = async () => {
        setLoading(true);
        await fetchClassificationCountData();
        const resj = await fetchTweet(token);

        if (!resj) {
            setTweet({
                tweetId: '',
                content: "Error connecting to server!"
            });
        } else if (resj.error) {
            setTweet({
                tweetId: '',
                content: resj.error
            });
            setDoneTagging(true);
        } else {
            setTweet({
                tweetId: resj.id,
                content: resj.content,
                tweetURL: resj.tweet_url,
            });
            setDoneTagging(false);
            setStartTime(Date.now());
        }

        setLoading(false);
    };

    const loadFeatures = async () => {
        const resj = await fetchFeatures();
        setFeaturesList(resj.map(feature => feature[1]));
    };

    // Action handlers
    const submitTweetTagging = async (classification, features) => {
        if (!tweet.tweetId) return;

        setLoading(true);

        const elapsedTime = (Date.now() - startTime) / 1000;
        // tagging duration above 6 minutes is considered an anomaly and will be ignored in average calculation
        // const maxTime = 360; // 6 minutes
        // const validTime = elapsedTime < maxTime ? elapsedTime : null;

        const success = await submitClassification(token, tweet.tweetId, classification, features, elapsedTime);
        if (success) {
            toast.success("Classification submitted!", { autoClose: 2000 });
            setSelectedFeatures([]);
            setStartTime(null);
            await getNewTweet();
        } else {
            toast.error("Error submitting classification.");
        }

        setLoading(false);
        setIsAntisemitic(false);
    };

    const signOut = () => {
        setPasscode(null);
        setToken(null);
    };

    const handlePanelToggle = (isPanelAdmin) => {
        if (isPanelAdmin) {
            setIsAdminPanelOpen(true);
            setIsUserPanelOpen(false);
        } else {
            setIsUserPanelOpen(true);
            setIsAdminPanelOpen(false);
        }
    };

    const closeAllPanels = () => {
        setIsUserPanelOpen(false);
        setIsAdminPanelOpen(false);
    };

    // Initialize data on component mount
    useEffect(() => {
        getNewTweet();
        loadFeatures();
        fetchClassificationCountData();
    }, []);

    // Component render
    return (
        <div id="form-frame">

            <h1 className="form-title">{isPro ? "Tweet Classifier - Pro" : " Tweet Classifier"}</h1>

            {/* Tweet Display Section */}
            {tweet.tweetId ? (
                <div>
                    <Tweet tweet={tweet} isPro={isPro}></Tweet>

                    <div className="copy-to-clip-div">
                        <CopyToClipboard
                            text={tweet.content || ""}
                            onCopy={() => {
                                toast.success("Text copied to clipboard", { autoClose: 2000 });
                            }}
                        >
                            <button className="copy-to-clip-btn" type="button">
                                <span className="bi bi-clipboard"></span>
                                <span style={{ paddingLeft: "5%" }}></span>
                                Copy
                            </button>
                        </CopyToClipboard>

                        {isPro && tweet.tweetURL && (
                            <button
                                className="tweet-link-btn"
                                type="button"
                                onClick={() => window.open(tweet.tweetURL, "_blank")}
                            >
                                Tweet Link
                                <span style={{ paddingRight: "5%" }}></span>
                                <span className="bi bi-box-arrow-up-right"></span>
                            </button>
                        )}
                    </div>
                </div>
            ) : (
                <div style={{textAlign: "center"}}>{tweet.content}</div>
            )}

            {/* Classification Toggle Switch */}
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

            {/* Classification Action Buttons */}
            <div className="classification-zone">
                <button
                    type="submit"
                    id="classify-btn"
                    className="submit-button classify-submit-button"
                    onClick={() => submitTweetTagging(isAntisemitic ? "Positive" : "Negative", selectedFeatures)}
                    disabled={loading || doneTagging || (isAntisemitic && selectedFeatures.length === 0)}
                >
                    Classify As {isAntisemitic ? "" : "Not "}Antisemitic
                </button>

                <button
                    id="not-sure-btn"
                    className="small-side-button"
                    type="button"
                    disabled={loading || doneTagging || isAntisemitic || isPro}
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
                    onClick={() => submitTweetTagging("Irrelevant", [])}
                    disabled={loading || doneTagging || isAntisemitic}
                >
                    <span>Irrelevant</span>
                    <span style={{paddingLeft: "3%"}}/>
                    <span className="bi bi-trash-fill"/>
                </button>
            </div>

            {/* Features Selection Section */}
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

            {/* Classification Counter */}
            {!isPro && (
                <div style={{ textAlign: "center" }}>
                    You've tagged so far {classificationCount} out of 125 tweets for this tagging round
                </div>
            )
            }


            {/* Footer Controls */}
            <div className="bottom-container">

                <button
                    id="sign-out-btn"
                    className="bottom-container-button"
                    type="button"
                    onClick={signOut}
                >
                    <span>Sign Out</span>
                    <span style={{ paddingRight: "3%" }} />
                    <span className="bi bi-door-closed" />
                </button>

                {isPro && (
                    <button
                        id="panel-btn"
                        className="bottom-container-button"
                        type="button"
                        onClick={() => handlePanelToggle(true)}
                    >
                        <span>{"Admin Panel"}</span>
                        <span style={{paddingLeft: "5%"}}/>
                        <span className="bi bi-clipboard-data-fill"/>
                    </button>
                )}

                <button
                    id="panel-btn"
                    className="bottom-container-button"
                    type="button"
                    onClick={() => handlePanelToggle(false)}
                >
                    <span>{"Profile"}</span>
                    <span style={{ paddingLeft: "5%" }} />
                    <span className="bi bi-person-circle" />
                </button>

                <button
                    id="spreadsheet-btn"
                    className="bottom-container-button"
                    type="button"
                    onClick={() => window.open("https://docs.google.com/spreadsheets/d/1qN6PNEhF44HznXzblx7JSLeRrmec5tTQZ-FIqBzZycE/edit?gid=1537961622#gid=1537961622")}
                >
                    Spreadsheet
                    <span style={{ paddingRight: "5%" }}></span>
                    <span className="bi bi-file-earmark-spreadsheet"></span>
                </button>


            </div>

            {/* User/Admin Stats Panel */}
            {(isUserPanelOpen || isAdminPanelOpen) && (
                <Panel
                    token={token}
                    showAdminPanel={isAdminPanelOpen}
                    onClose={closeAllPanels}
                />
            )}

            <ToastContainer/>
        </div>
    );
};

export default MainView;