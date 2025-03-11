import { useEffect, useState } from "react";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "react-tooltip/dist/react-tooltip.css";
import "./MainView.css";
import Tweet from "./Tweet";
import FeatureButton from "./FeatureButton";
import Panel from "./Panels/Panel";
import { CopyToClipboard } from "react-copy-to-clipboard";

const MainView = ({ passcode, isPro, setPasscode, token, setToken }) => {
  const [isAntiMil, setIsAntiMil] = useState(false);
  const [isFinished, setIsFinished] = useState(true);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [copyStatus, setCopyStatus] = useState(false);

  const onCopyText = () => {
    setCopyStatus(true);
    setTimeout(() => setCopyStatus(false), 2000);
  };

  const [tweet, setTweet] = useState({
    tweetId: "",
    tweetText: "",
    username: "",
    tweetURL: "",
  });
  const [classifiyCount, setClassifyCount] = useState(0);
  const [featuresDetails, setFeaturesDetails] = useState([]);

  // Enable tooltips
  useEffect(() => {
    const tooltipTriggerList = document.querySelectorAll('[data-toggle="tooltip"]');
    tooltipTriggerList.forEach(function (tooltipTriggerEl) {
      new window.bootstrap.Tooltip(tooltipTriggerEl);
    });
  }, []);

  // Get features on component mount
  useEffect(() => {
    if (featuresDetails.length > 0) {
      return;
    }
    fetch(window.API_URL + "/params_list", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    }).then((response) => {
      if (response.ok) {
        response.json().then((resj) => {
          const featuresDetails_l = resj.map((feature) => {
            return {
              key: feature[0],
              description: feature[1],
              checked: false,
            };
          });
          setFeaturesDetails(featuresDetails_l);
        });
      }
    });
  }, []);

  const updateCount = () => {
    fetch(window.API_URL + "/count_classifications", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + token,
      },
    }).then((response) => {
      if (response.ok) {
        response.json().then((resj) => {
          setClassifyCount(resj.count);
        });
      }
    });
  };

  const signOut = () => {
    setPasscode(null);
    setToken(null);
  };

  const getNewTweet = async () => {
    // Get new tweet from server and set tweet state
    fetch(window.API_URL + "/get_tweet", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + token,
      },
    }).then((response) => {
      // If response is ok, get new tweet and set tweet state
      if (response.ok) {
        response.json().then((resj) => {
          if (resj.error) {
            const errorTweet = {
              tweetId: "ERROR",
              tweetText: resj.error === "No unclassified tweets" ? "Hurray! All tweets classified!" : "Error, Please contact support.",
              username: "Guest",
            };
            setIsFinished(true);
            setTweet(errorTweet);
            toast.success(errorTweet.tweetText, {
              position: toast.POSITION.TOP_RIGHT,
              autoClose: 2000,
            });
            return;
          }
          // Create tweet object
          const tweet_l = {
            tweetId: resj.id,
            tweetText: resj.content,
            username: resj.tweeter,
            tweetURL: resj.tweet_url,
          };
          // Set tweet state
          setIsFinished(false);
          setTweet(tweet_l);
        });
      }
    });
  };

  useEffect(() => {
    let isMounted = true;

    const fetchData = async () => {
      try {
        const response = await fetch(window.API_URL + "/get_tweet", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: "Bearer " + token,
          },
        });
        if (response.ok && isMounted) {
          const resj = await response.json();
          if (resj.error) {
            // Handle error cases
            const errorTweet = {
              tweetId: "ERROR",
              tweetText: resj.error === "No unclassified tweets" ? "Hurray! All tweets classified!" : "Error, Please contact support.",
              username: "Guest",
            };
            setIsFinished(true);
            setTweet(errorTweet);
            toast.success(errorTweet.tweetText, {
              position: toast.POSITION.TOP_RIGHT,
              autoClose: 2000,
            });
          } else {
            const tweet_l = {
              tweetId: resj.id,
              tweetText: resj.content,
              username: resj.tweeter,
              tweetURL: resj.tweet_url,
            };
            setIsFinished(false);
            setTweet(tweet_l);
          }
        }
      } catch (error) {
        console.error("Error fetching new tweet:", error);
        // Handle fetch error
        const errorTweet = {
          tweetId: "ERROR",
          tweetText: "Error, Please contact support.",
          username: "Guest",
        };
        setIsFinished(true);
        setTweet(errorTweet);
        toast.error("Error getting new tweet", {
          position: toast.POSITION.TOP_RIGHT,
          autoClose: 2000,
        });
      }
    };

    fetchData();
    updateCount();

    return () => {
      // Cleanup function to prevent state update on unmounted component
      isMounted = false;
    };
  }, [token]);

  const resetFeatures = () => {
    // Reset features
    const featuresDetails_l = featuresDetails.map((feature) => {
      return {
        key: feature.key,
        description: feature.description,
        checked: false,
      };
    });
    setFeaturesDetails(featuresDetails_l);
  };

  const submitClassification = async (classification, features_l) => {
    // Send classification to server
    fetch(window.API_URL + "/classify_tweet", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + token,
      },
      body: JSON.stringify({
        tweet_id: tweet.tweetId,
        classification: classification,
        features: features_l,
      }),
    })
      .then((response) => {
        if (response.ok) {
          response.json().then((resj) => {
            if (!resj.error) {
              toast.success("Classification sent successfully", {
                position: toast.POSITION.TOP_RIGHT,
                autoClose: 2000,
              });
            }
          });

          // // Get new tweet
          // getNewTweet();
          // // Update classification count
          // updateCount();
        } else {
          toast.error("Error sending classification", {
            position: toast.POSITION.TOP_RIGHT,
            autoClose: 2000,
          });
        }
      })
      .finally(() => {
        // Get new tweet
        getNewTweet();
        // Update classification count
        updateCount();
      });
  };

  const clearForm = () => {
    // Clear form
    document.getElementById("flexSwitchCheckDefault").checked = false;
    setIsAntiMil(false);
    // Reset features
    resetFeatures();
  };

  const notSureClick = () => {
    // Send a not sure classification to server
    submitClassification("Unknown", null);
    // Clear form
    clearForm();
  };

  const irrelevantClick = () => {
    // Send an irrelevant classification to server
    submitClassification("Irrelevant", null);

    // Clear form
    clearForm();
  };

  const skipTweet = async () => {
    // Fetch a new tweet from the server and update the tweet state
    getSkipTweet();
  };

  const getSkipTweet = async () => {
    // Get new tweet from server and set tweet state

    fetch(window.API_URL + "/get_skip_tweet", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + token,
      },
      body: JSON.stringify({
        curr_id: tweet.tweetId,
      }),
    }).then((response) => {
      // If response is ok, get new tweet and set tweet state
      if (response.ok) {
        response.json().then((resj) => {
          if (resj.error) {
            const errorTweet = {
              tweetId: "ERROR",
              tweetText: resj.error === "No unclassified tweets" ? "Hurray! All tweets classified!" : "Error, Please contact support.",
              username: "Guest",
            };
            setIsFinished(true);
            setTweet(errorTweet);
            toast.success(errorTweet.tweetText, {
              position: toast.POSITION.TOP_RIGHT,
              autoClose: 2000,
            });
            return;
          } else if (resj.id === tweet.tweetId) {
            // Check if returned tweet equals to current tweet_id.
            toast.error("Cannot skip current tweet", {
              position: toast.POSITION.TOP_RIGHT,
              autoClose: 2000,
            });
          } else {
            // Otherwise, create a tweet.
            const tweet_l = {
              tweetId: resj.id,
              tweetText: resj.content,
              username: resj.tweeter,
            };
            // Set tweet state
            setIsFinished(false);
            setTweet(tweet_l);
          }
        });
      }
    });
  };

  const getFeaturesObj = () => {
    // Get checked features
    const features_l = featuresDetails.filter((feature) => feature.checked);
    // Get all features keys
    const featuresKeys = featuresDetails.map((feature) => feature.key);
    // Create features object
    const featuresObj = {};
    featuresKeys.forEach((key) => {
      featuresObj[key] = false;
    });
    // Set features to true
    features_l.forEach((feature) => {
      featuresObj[feature.key] = true;
    });
    return featuresObj;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const isAntisemitic_l = document.getElementById("flexSwitchCheckDefault").checked;
    const classification = isAntisemitic_l ? "Positive" : "Negative";
    const featuresObj = isAntisemitic_l ? getFeaturesObj() : null;

    // Clear form
    clearForm();

    // Send classification to server
    submitClassification(classification, featuresObj);
  };

  const setFeature = (feature, bool) => {
    setFeaturesDetails((prevFeatures) => {
      const index = prevFeatures.findIndex((f) => f.key === feature.key);
      const updatedFeatures = [...prevFeatures];
      updatedFeatures[index] = {
        ...updatedFeatures[index],
        checked: bool,
      };
      return updatedFeatures;
    });
  };

  const closePanel = () => {
    setIsPanelOpen(false);
  };

  const handlePanelButtonClick = () => {
    setIsPanelOpen(true);
  };

  return (
    <>
      <div id="form-frame">
        <h1 className="form-title">{isPro ? "S&O Classifier - Pro" : "S&O Classifier"}</h1>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <Tweet tweet={tweet} isPro={isPro} />

            <div className="copy-to-clip-div">
              <CopyToClipboard text={tweet.tweetText} onCopy={onCopyText}>
                <button id="copy-to-clip-btn" className="copy-to-clip-btn" type="button">
                  <span className="bi bi-clipboard" />
                  <span style={{ paddingLeft: "5%" }} />
                  Copy
                </button>
              </CopyToClipboard>
              <div className={`copy-status ${copyStatus ? "show" : ""}`}>Text copied to clipboard!</div>

              <button disabled={isFinished} id="skip-tweet-btn" className="skip-tweet-btn" type="button" onClick={skipTweet}>
                Skip
                <span className="bi bi-skip-forward-fill ms-2" />
              </button>
            </div>

            <p className="classify-count">Classifications made by your account: {classifiyCount}</p>

            <br />
            <div className="form-check form-switch form-switch-md">
              <input
                className="form-check-input"
                type="checkbox"
                disabled={isFinished}
                id="flexSwitchCheckDefault"
                onChange={() => setIsAntiMil(!isAntiMil)}
              />

              <label className="form-check-label" htmlFor="flexSwitchCheckDefault">
                {isAntiMil ? (
                  <span>
                    <strong>Antisemitic</strong>
                  </span>
                ) : (
                  <span>Not Antisemitic</span>
                )}
              </label>
            </div>
          </div>
          {isAntiMil ? (
            <div className="features-container">
              {featuresDetails.map((feature, index) => (
                <FeatureButton key={index} feature={feature} disabled={!isAntiMil} setFeature={setFeature} />
              ))}
            </div>
          ) : null}
          <div className="classification-zone">
            <div>
              <div className="classify-container">
                <button type="submit" id="classify-btn" disabled={isFinished} className="submit-button classify-submit-button">
                  Classify As {isAntiMil ? "" : "Not"} Antisemitic
                </button>
              </div>
              <button id="not-sure-btn" className="small-side-button" disabled={isFinished} type="button" onClick={notSureClick}>
                <span className="bi bi-question-octagon-fill" />
                <span style={{ paddingLeft: "3%" }} />
                <span>Uncertain</span>
              </button>
              <span style={{ paddingLeft: "2%" }} />
              <button id="irrelevant-btn" className="small-side-button" disabled={isFinished} type="button" onClick={irrelevantClick}>
                <span>Irrelevant</span>
                <span style={{ paddingLeft: "3%" }} />
                <span className="bi bi-trash-fill" />
              </button>
            </div>
          </div>
        </form>
        <div className="bottom-container">
          <button id="sign-out-btn" className="bottom-container-button" type="button" onClick={signOut}>
            <span className="bi bi-door-closed" />
            <span style={{ paddingLeft: "3%" }} />
            <span>Sign Out</span>
          </button>

          <button id="panel-btn" className="bottom-container-button" type="button" onClick={handlePanelButtonClick}>
            <span>{isPro ? "Admin Panel" : "Profile"}</span>
            <span style={{ paddingLeft: "5%" }} />
            <span className="bi bi-person-circle" />
          </button>
          {isPanelOpen ? <Panel token={token} passcode={passcode} onClose={closePanel} isPro={isPro} /> : ""}
        </div>
        {/* TODO: check if a break line is needed */}
        {/* <br/> */}
      </div>
      <ToastContainer />
    </>
  );
};

export default MainView;
