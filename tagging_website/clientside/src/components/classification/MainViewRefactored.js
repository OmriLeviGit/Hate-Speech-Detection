import { useEffect, useState } from "react";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const MainViewRefactored = ({ token }) => {

    // State for managing the tweet
    const [tweet, setTweet] = useState(null);
    const [loading, setLoading] = useState(false);
    // Stores the selected features of a tweet before submission, if is being tagged "Positive"
    const [selectedFeatures, setSelectedFeatures] = useState([]);


    // Sends a request to the server to fetch a tweet to tag
    const getNewTweet = async () => {
        setLoading(true);  // Show loading state
        try {
            const response = await fetch(window.API_URL + "/get_tweet_to_tag", {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + token,
                },
            });

            if (!response.ok) {
                throw new Error("Failed to fetch tweet.");
            }

            const resj = await response.json();

            if (resj.error) {
                setTweet({ content: "No tweets left to classify!" });
            } else {
                setTweet({
                    tweetId: resj.id,
                    content: resj.content,
                    tweetURL: resj.tweet_url,
                });
            }
        } catch (error) {
            console.error("Error fetching tweet:", error);
            setTweet({ content: "Error connecting to server!" });
        } finally {
            setLoading(false);  // Remove loading state
        }
    };


    // Sends a request to the server to tag the current tweet
    const submitTweetTagging = async (classification, features) => {
        if (!tweet) return; // No tweet loaded

        console.log("Submitting classification:", {
            tweet_id: tweet.tweetId,
            classification: classification,
            features: features,
        });

        try {
            const response = await fetch(window.API_URL + "/submit_tweet_tag", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + token,
                },
                body: JSON.stringify({
                    tweet_id: tweet.tweetId,
                    classification: classification,
                    features: features,
                }),
            });

            if (!response.ok) {
                throw new Error("Failed to submit classification.");
            }

            toast.success("Classification submitted!", { autoClose: 2000 });

            // Fetch a new tweet after submitting
            await getNewTweet();

        } catch (error) {
            console.error("Error submitting classification:", error);
            toast.error("Error submitting classification.");
        }
    };



    // Fetch a tweet on component mount
    useEffect(() => {
        getNewTweet();
    }, []);


    return (
        <div className="main-view">
            <h1>Tweet Classification</h1>
            {loading ? <p>Loading...</p> : <p>{tweet ? tweet.content : "No tweet loaded yet."}</p>}
            <div>
                <h3>Select Features:</h3>
                {["Bias", "Hate Speech", "Disinformation"].map((feature) => (
                    <label key={feature}>
                        <input
                            type="checkbox"
                            value={feature}
                            checked={selectedFeatures.includes(feature)}
                            onChange={(e) => {
                                if (e.target.checked) {
                                    setSelectedFeatures([...selectedFeatures, feature]);
                                } else {
                                    setSelectedFeatures(selectedFeatures.filter(f => f !== feature));
                                }
                            }}
                        />
                        {feature}
                    </label>
                ))}
            </div>

            <div>
                <button onClick={() => submitTweetTagging("Positive", selectedFeatures)} disabled={loading}>
                    Positive
                </button>
                <button onClick={() => submitTweetTagging("Negative", selectedFeatures)} disabled={loading}>
                    Negative
                </button>
                <button onClick={() => submitTweetTagging("Irrelevant", selectedFeatures)} disabled={loading}>
                    Irrelevant
                </button>
            </div>

            <ToastContainer />
        </div>
    );
};

export default MainViewRefactored;
