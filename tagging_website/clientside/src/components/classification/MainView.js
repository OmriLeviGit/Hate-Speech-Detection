import {useEffect, useState} from 'react';
import {toast, ToastContainer} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import 'react-tooltip/dist/react-tooltip.css';
import './MainView.css';
import Tweet from './Tweet';
import FeatureButton from './FeatureButton';
import Panel from './Panels/Panel';
import {CopyToClipboard} from 'react-copy-to-clipboard';


const MainView = ({passcode, isPro, setPasscode, token, setToken}) => {
  const [isAntisemitic, setIsAntisemitic] = useState(false);
  const [isFinished, setIsFinished] = useState(true);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [copyStatus, setCopyStatus] = useState(false);

  const onCopyText = () => {
    setCopyStatus(true);
    setTimeout(() => setCopyStatus(false), 2000);
  };

  const [tweet, setTweet] = useState({
    tweetId: '',
    tweetText: '',
    username: '',
    tweetURL: '',
  });
  const [classifyCount, setClassifyCount] = useState(0);
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
    fetch(window.API_URL + '/params_list', {
      method: "GET", headers: {
        "Content-Type": "application/json",
      }
    }).then(response => {
      if (response.ok) {
        response.json().then(resj => {
          const featuresDetails_l = resj.map(feature => {
            return {
              key: feature[0],
              description: feature[1],
              checked: false,
            }
          });
          setFeaturesDetails(featuresDetails_l);
        });
      }
    });
  }, []);

  const updateCount = () => {
    fetch(window.API_URL + '/count_tags_made_by_user', {
      method: "GET", headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token,
      }
    }).then(response => {
      if (response.ok) {
        response.json().then(resj => {
          setClassifyCount(resj.count);
        });
      }
    });
  }

  const signOut = () => {
    setPasscode(null);
    setToken(null);
  }

  const getNewTweet = async () => {
    // Get new tweet from server and set tweet state
    fetch(window.API_URL + '/get_tweet_to_tag', {
      method: "GET", headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token,
      }
    }).then(response => {
      // If response is ok, get new tweet and set tweet state
      if (response.ok) {
        response.json().then(resj => {
          if (resj.error) {
            const errorTweet = {
              tweetId: 'ERROR',
              tweetText: resj.error === 'No unclassified tweets'
                  ? 'Hurray! All tweets classified!'
                  : 'Error, Please contact support.',
              username: 'Guest',
            };
            setIsFinished(true);
            setTweet(errorTweet);
            toast.success(errorTweet.tweetText, {
              position: toast.POSITION.TOP_RIGHT,
              autoClose: 2000
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
  }

  // Updates whenever a token has changed
  useEffect(() => {
    let isMounted = true;
    const fetchData = async () => {
      try {
        const response = await fetch(window.API_URL + '/get_tweet_to_tag', {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
          }
        });
        if (response.ok && isMounted) {
          const resj = await response.json();
          if (resj.error) {
            // Handle error cases
            const errorTweet = {
              tweetId: 'ERROR',
              tweetText: resj.error === 'No unclassified tweets'
                  ? 'Hurray! All tweets classified!'
                  : 'Error, Please contact support.',
              username: 'Guest',
            };
            setIsFinished(true);
            setTweet(errorTweet);
            toast.success(errorTweet.tweetText, {
              position: toast.POSITION.TOP_RIGHT,
              autoClose: 2000
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
          tweetId: 'ERROR',
          tweetText: 'Error, Please contact support.',
          username: 'Guest',
        };
        setIsFinished(true);
        setTweet(errorTweet);
        toast.error("Error getting new tweet", {
          position: toast.POSITION.TOP_RIGHT,
          autoClose: 2000
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


  // Unselects all selected tagging features
  const resetFeatures = () => {
    const featuresDetails_l = featuresDetails.map(feature => {
      return {
        key: feature.key,
        description: feature.description,
        checked: false,
      }
    });
    setFeaturesDetails(featuresDetails_l);
  }

  // Submits a tagging of a tweet
  const submitTweetTagging = async (classification, features_l) => {

    console.log("submitTweetTagging - Submitting Request:", {
      tweet_id: tweet.tweetId,
      classification: classification,
      features: features_l,  // Make sure this is an array!
    });

    // Send the tagging request to the server
    fetch(window.API_URL + '/submit_tweet_tag', {
      method: "POST", headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token,
      }, body: JSON.stringify(
          {
            tweet_id: tweet.tweetId,
            classification: classification,
            features: features_l,
            // The next line is commented due to a potential mismatch with the backend base_models.py file
            // features: JSON.stringify(features_l),
          })
    }).then(response => {
      if (response.ok) {
        response.json().then(resj => {
          if (!resj.error) {
            toast.success("Tag Sent Successfully", {
              position: toast.POSITION.TOP_RIGHT,
              autoClose: 2000
            });
          }
        });
        // Get new tweet
        getNewTweet();
        // Update classification count
        updateCount();

      } else {
        toast.error("Error Sending Tag", {
          position: toast.POSITION.TOP_RIGHT,
          autoClose: 2000
        });
      }
    }).finally(() => {
      // Get new tweet
      getNewTweet();
      // Update classification count
      updateCount();
    });
  }

  // Clears form
  const clearForm = () => {
    document.getElementById("flexSwitchCheckDefault").checked = false;
    setIsAntisemitic(false);
    // Reset features
    resetFeatures();
  }

  const uncertainClick = () => {
    // Send a not sure classification to server
    submitTweetTagging("Uncertain", null);
    // Clear form
    clearForm();
  }


  const irrelevantClick = () => {
    // Send an irrelevant classification to server
    submitTweetTagging("Irrelevant", null);

    // Clear form
    clearForm();
  }


  const getFeaturesList = () => {
    // Convert the object into a list of selected feature descriptions
    return featuresDetails
        .filter(feature => feature.checked)  // Keep only selected features
        .map(feature => feature.description);  // Extract only human-readable names
  };



  const handleSubmit = async (e) => {
    e.preventDefault();
    const isAntisemitic_l = document.getElementById("flexSwitchCheckDefault").checked;
    const classification = isAntisemitic_l ? "Positive" : "Negative";
    const featuresList = isAntisemitic_l ? getFeaturesList() : null;

    // Clear form
    clearForm();

    // Send classification to server
    submitTweetTagging(classification, featuresList);
  }

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

              <Tweet tweet={tweet} isPro={isPro}/>

              <div className='copy-to-clip-div'>
                <CopyToClipboard text={tweet.tweetText} onCopy={onCopyText}>
                  <button id="copy-to-clip-btn" className="copy-to-clip-btn" type="button">
                    <span className="bi bi-clipboard"/>
                    <span style={{paddingLeft: "5%"}}/>
                    Copy
                  </button>
                </CopyToClipboard>
                <div className={`copy-status ${copyStatus ? "show" : ''}`}>Text copied to clipboard!</div>
              </div>

              <p className="classify-count">Classifications made by your account: {classifyCount}</p>

              <br/>

              <div className="form-check form-switch form-switch-md">
                <input className="form-check-input" type="checkbox" disabled={isFinished}
                       id="flexSwitchCheckDefault"
                       onChange={() => setIsAntisemitic(!isAntisemitic)}/>

                <label className="form-check-label" htmlFor="flexSwitchCheckDefault">
                  {isAntisemitic ? <span><strong>Antisemitic</strong></span> :
                      <span>Not Antisemitic</span>}
                </label>
              </div>
            </div>
            {isAntisemitic ? (
                <div className="features-container">
                  {featuresDetails.map((feature, index) => (
                      <FeatureButton
                          key={index}
                          feature={feature}
                          disabled={!isAntisemitic}
                          setFeature={setFeature}
                      />
                  ))}
                </div>
            ) : null}


            <div className="classification-zone">
              <div>
                <div className="classify-container">
                  <button
                      type="submit" id="classify-btn"
                      disabled={isFinished}
                      className="submit-button classify-submit-button"
                  >Classify As {isAntisemitic ? '' : 'Not'} Antisemitic
                  </button>
                </div>

                <button id="not-sure-btn"
                        className="small-side-button"
                        disabled={isFinished}
                        type="button"
                        onClick={uncertainClick}>
                  <span className="bi bi-question-octagon-fill"/>
                  <span style={{paddingLeft: "3%"}}/>
                  <span>Uncertain</span>
                </button>

                <span style={{paddingLeft: "2%"}}/>
                <button id="irrelevant-btn"
                        className="small-side-button"
                        disabled={isFinished}
                        type="button"
                        onClick={irrelevantClick}>
                  <span>Irrelevant</span>
                  <span style={{paddingLeft: "3%"}}/>
                  <span className="bi bi-trash-fill"/>
                </button>
              </div>
            </div>
          </form>
          <div className="bottom-container">
            <button id="sign-out-btn"
                    className="bottom-container-button"
                    type="button"
                    onClick={signOut}>
              <span className="bi bi-door-closed"/>
              <span style={{paddingLeft: "3%"}}/>
              <span>Sign Out</span>
            </button>

            <button id="panel-btn"
                    className="bottom-container-button"
                    type="button"
                    onClick={handlePanelButtonClick}>
              <span>{isPro ? "Admin Panel" : "Profile"}</span>
              <span style={{paddingLeft: "5%"}}/>
              <span className="bi bi-person-circle"/>
            </button>
            {isPanelOpen ?
                <Panel token={token} passcode={passcode} onClose={closePanel} isPro={isPro}/>
                : ""}
          </div>
          {/* TODO: check if a break line is needed */}
          {/* <br/> */}

        </div>
        <ToastContainer/>
      </>
  );
}

export default MainView;