import { useEffect, useState } from "react";
import { toast, ToastContainer } from "react-toastify";
import { fetchTweet, submitClassification, fetchFeatures } from "../../services/api";
import "react-toastify/dist/ReactToastify.css";

// ToDo - Load the feature params from the server
// ToDo - Enable selecting features only when tagging as positive
// ToDo - Show classifications count to the user
// ToDo - Add a sign out button
// ToDo - Add profile button
// ToDo - Show the personal statistics
// ToDo - right the todos for Pro users

const MainViewRefactored = ({ token }) => {

    // State for managing the tweet
    const [tweet, setTweet] = useState(null);
    // Disables the buttons while sending a tag submission
    const [loading, setLoading] = useState(false);
    // Stores the feature list
    const [featuresList, setFeaturesList] = useState([]);
    // Stores the selected features of a tweet before submission, if is being tagged "Positive"
    const [selectedFeatures, setSelectedFeatures] = useState([]);


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
            setTweet({ content: "No tweets left to classify! 🎉" });
        } else {
            setTweet({
                tweetId: resj.id,
                content: resj.content,
                tweetURL: resj.tweet_url,
            });
        }

        setLoading(false);
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
    };




    // Fetch a tweet on component mount
    useEffect(() => {
        getNewTweet();
        loadFeatures();
    }, []);


    return (
        <div className="main-view">
            <h1>Tweet Classification</h1>
            {loading ? <p>Loading...</p> :
                <p>{tweet ? (
                    <p>{tweet.content}</p>
                ) : (
                    <p style={{ color: "gray", fontStyle: "italic" }}>🎉 All tweets are classified!</p>
                )}
                </p>}

            <div>
                <h3>Select Features:</h3>
                {featuresList.map((feature) => (
                    <label key={feature}>
                        <input
                            type="checkbox"
                            value={feature}
                            checked={selectedFeatures.includes(feature)}
                            onChange={() => handleFeatureSelection(feature)}
                        />
                        {feature}
                    </label>
                ))}
            </div>



            <div>
                <button onClick={() => submitTweetTagging("Positive", selectedFeatures)}
                        disabled={loading || !tweet || selectedFeatures.length === 0}>
                    Positive
                </button>
                <button onClick={() => submitTweetTagging("Negative", selectedFeatures)} disabled={loading || !tweet}>
                    Negative
                </button>
                <button onClick={() => submitTweetTagging("Irrelevant", selectedFeatures)} disabled={loading || !tweet}>
                    Irrelevant
                </button>
            </div>


            <ToastContainer />
        </div>
    );
};

export default MainViewRefactored;
