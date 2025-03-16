import './MainViewRefactored.css';
import { useEffect, useState } from "react";
import { toast, ToastContainer } from "react-toastify";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { fetchTweet, submitClassification, fetchFeatures, fetchClassificationCount, fetchUserStats  } from "../../services/api";
import FeatureButton from './FeatureButton';
import Panel from "./Panels/Panel";
import "react-toastify/dist/ReactToastify.css";
import Tweet from "./Tweet";

// ToDo - Show the personal statistics when clicking the profile button
// ToDo - For Pro users:
//  - Keep Profile button for pro for pro user personal stats
//  - Add a User Stats button to show them all users stats
//  - Add download csv for users stats
//  - Attach a link to the original tweet

const MainViewRefactored = ({ token, setToken, passcode, setPasscode, isPro }) => {

    // Helps the toggle button of tagging as antisemitic or not antisemitic
    const [isAntisemitic, setIsAntisemitic] = useState(false);

    // Helps with showing/hiding the personal user stats panel
    const [isUserPanelOpen, setIsUserPanelOpen] = useState(false);

    // Helps with showing/hiding the pro (admin) stats panel
    const [isAdminPanelOpen, setIsAdminPanelOpen] = useState(false);

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
            setTweet({
                tweetId: '',
                content: "No tweets left to classify! 🎉" });
            setDoneTagging(true);

        } else {
            setTweet({
                tweetId: resj.id,
                content: resj.content,
                tweetURL: resj.tweet_url,

            });
            setDoneTagging(false);
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

            {tweet && tweet.tweetId ?
                <div>

                    <Tweet tweet={tweet} isPro={isPro}></Tweet>

                    <div className="copy-to-clip-div">
                        <CopyToClipboard text={tweet ? tweet.content : ""} onCopy={() => {
                            toast.success("Text copied to clipboard", { autoClose: 2000 });
                        }}>
                            <button
                                className="copy-to-clip-btn"
                                type="button">
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
                : <div style={{textAlign: "center" }}>All tweets are classified! 🎉</div>
            }


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


            {/* The next div has the "Classify As Antisemitic/Not Antisemitic", "Uncertain" and "Irrelevant" buttons */}
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
                    onClick={() => submitTweetTagging("Uncertain", [])}
                    disabled={loading || doneTagging || isAntisemitic}
                >
                    <span className="bi bi-question-octagon-fill"/>
                    <span style={{paddingLeft: "3%"}}/>
                    <span>Uncertain</span>
                </button>

                {/* ToDo - When a pro user is logged-in, consider remove "Uncertain" button instead of just disabling it and also make "Irrelevant" button spread like the classify button above
                  */}

                <span style={{paddingLeft: "2%"}}/>

                <button
                    id="irrelevant-btn"
                    className="small-side-button"
                    type="button"
                    disabled={loading || doneTagging || isAntisemitic || isPro}
                    onClick={() => submitTweetTagging("Irrelevant", [])}
                >
                    <span>Irrelevant</span>
                    <span style={{paddingLeft: "3%"}}/>
                    <span className="bi bi-trash-fill"/>
                </button>

            </div>

            {/* If the user has decided to tag as antisemitic, it opens the features list */}
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


            <div>
                Classifications made by your account: {classificationCount}
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

                {/* Admin panel button - clicking it opens all user stats */}
                {isPro ?
                    <button id="panel-btn"
                            className="bottom-container-button"
                            type="button"
                            onClick={() => {
                                setIsAdminPanelOpen(true);
                                setIsUserPanelOpen(false);  // Close the other panel
                            }
                            }>
                        <span>{"Admin Panel"}</span>
                        <span style={{paddingLeft: "5%"}}/>
                        <span className="bi bi-clipboard-data-fill"/>
                    </button>
                    :
                    <>
                    </>
                }

                {/* Profile button - clicking it opens current logged-in user stats */}
                <button id="panel-btn"
                        className="bottom-container-button"
                        type="button"
                        onClick={() =>{
                            setIsUserPanelOpen(true)
                            setIsAdminPanelOpen(false);  // Close the other panel
                        }
                        }>
                    <span>{"Profile"}</span>
                    <span style={{ paddingLeft: "5%" }} />
                    <span className="bi bi-person-circle" />
                </button>

            </div>

            {(isUserPanelOpen || isAdminPanelOpen) && (
                <Panel
                    token={token}
                    showAdminPanel={isAdminPanelOpen}  // Determines which panel to show
                    onClose={() => {
                        setIsUserPanelOpen(false);
                        setIsAdminPanelOpen(false);
                    }}
                />
            )}

            <ToastContainer/>
        </div>
    );
};

export default MainViewRefactored;
