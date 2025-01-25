import './Tweet.css';

const Tweet = ({ tweet }) => {
    return (
        <div className="tweet-div">
            <p className="tweet-text" dir='auto' > {tweet.tweetText} </p>
            <p className="tweet-id"> {tweet.tweetId === "ERROR" ? '' : tweet.tweetId}</p>
        </div>
    );

}

export default Tweet;