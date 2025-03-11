import { useState, useEffect } from "react";
import "./MainView.css";
import Tweet from "./Tweet";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const MainView = () => {
  const [isFinished, setIsFinished] = useState(false);
  const [isAntisemistic, setIsAntiMil] = useState(false);
  const [classifyCount, setClassifyCount] = useState(4);
  const [tweetNumber, setTweetNumber] = useState(35);
  const [tweet, setTweet] = useState({
    tweetId: `${classifyCount + 1}`,
    tweetText: `Sample tweet number ${tweetNumber}`,
    username: "TestUser",
  });

  // advance classification made
  useEffect(() => {
    if (isFinished) {
      // Increment classifyCount first
      setClassifyCount((prevCount) => {
        const newCount = prevCount + 1;
        // Create a new tweet based on updated classifyCount
        setTweet({
          tweetId: `${newCount + 1}`,
          tweetText: `Sample tweet number ${tweetNumber + 1}`,
          username: "TestUser",
        });
        return newCount; // Return the updated count
      });
      setIsFinished(false); // Reset isFinished to prevent repeat execution
      setTweetNumber((num) => num + 1);
    }
  }, [isFinished]);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Simulate classification submission
    toast.success("Classification sent successfully!");
    setIsFinished(true);
  };

  return (
    <div id="form-frame">
      <h1 className="form-title">Data Tagging</h1>
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
            <input
              className="form-check-input"
              type="checkbox"
              id="flexSwitchCheckDefault"
              onChange={() => setIsAntiMil(!isAntisemistic)}
            />
            <label className="form-check-label" htmlFor="flexSwitchCheckDefault">
              {isAntisemistic ? "Antisemistic" : "Not antisemistic"}
            </label>
          </div>
        </div>

        <div className="classification-zone">
          <button type="submit" id="classify-btn" className="submit-button classify-submit-button" disabled={isFinished}>
            Classify: {isAntisemistic ? "Antisemistic" : "Not antisemistic"}
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
