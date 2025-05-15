import './Tweet.css';

const Tweet = ({ tweet, isPro }) => {
    return (
            <div className="tweet-div">
                <p className="tweet-text" dir='auto' > {tweet.content} </p>

                {/* The Tweet ID tag is disabled as Eli asked to hide it from the taggers*/}
                {/*<p className="tweet-id"> {tweet.tweetId === "ERROR" ? '' : `Tweet ID: ${tweet.tweetId}`}</p>*/}
            </div>
    );

}

export default Tweet;