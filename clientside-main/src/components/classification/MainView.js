import { useState, useEffect } from "react";
import "./MainView.css";
import Tweet from "./Tweet";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const MainView = () => {
  const [tweet, setTweet] = useState({
    tweetId: "1",
    tweetText: "Sample tweet content without authentication.",
    username: "TestUser",
  });
  const [isFinished, setIsFinished] = useState(false);
  const [isAntiMil, setIsAntiMil] = useState(false);
  const [classifyCount, setClassifyCount] = useState(0);

  useEffect(() => {
    // Simulate API call, just set static data for testing purposes
    setTweet({
      tweetId: "1",
      tweetText: "Sample tweet content.",
      username: "TestUser",
    });
    setClassifyCount(10); // Example count
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Simulate classification submission
    toast.success("Classification sent successfully!");
    setIsFinished(true);
  };

  return (
    <div id="form-frame">
      <h1 className="form-title">S&O Classifier (No Authentication)</h1>
      <form onSubmit={handleSubmit}>
        <div className="copy-to-clip-div">
          <button type="button" className="copy-to-clip-btn">
            Copy
          </button>

          <button type="button" className="skip-tweet-btn" onClick={() => setIsFinished(true)}>
            Skip
          </button>
        </div>

        <div className="form-group">
          <Tweet tweet={tweet} />
          <p className="classify-count">Classifications made: {classifyCount}</p>
          <br />
          <div className="form-check form-switch form-switch-md">
            <input className="form-check-input" type="checkbox" id="flexSwitchCheckDefault" onChange={() => setIsAntiMil(!isAntiMil)} />
            <label className="form-check-label" htmlFor="flexSwitchCheckDefault">
              {isAntiMil ? "Anti Militias" : "Supports Militias"}
            </label>
          </div>
        </div>

        <div className="classification-zone">
          <button type="submit" id="classify-btn" className="submit-button classify-submit-button" disabled={isFinished}>
            Classify: {isAntiMil ? "Against Militias" : "In Favor of Militias"}
          </button>

          <button
            type="button"
            id="not-sure-btn"
            className="small-side-button"
            disabled={isFinished}
            onClick={() => toast.info("Classification not sure")}
          >
            Uncertain
          </button>

          <button
            type="button"
            id="irrelevant-btn"
            className="small-side-button"
            disabled={isFinished}
            onClick={() => toast.info("Classification marked as irrelevant")}
          >
            Irrelevant
          </button>
        </div>
      </form>

      <ToastContainer />
    </div>
  );
};

export default MainView;
