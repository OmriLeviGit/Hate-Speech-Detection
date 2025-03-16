export const fetchTweet = async (token) => {
    try {
        const response = await fetch(window.API_URL + "/get_tweet_to_tag", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token,
            },
        });

        return response.ok ? await response.json() : null;
    } catch (error) {
        console.error("Error fetching tweet:", error);
        return null;
    }
};

export const submitClassification = async (token, tweetId, classification, features, taggingDuration) => {
    try {
        const response = await fetch(window.API_URL + "/submit_tweet_tag", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token,
            },
            body: JSON.stringify({
                tweet_id: tweetId,
                classification,
                features,
                tagging_duration: taggingDuration
            }),
        });

        return response.ok;
    } catch (error) {
        console.error("Error submitting classification:", error);
        return false;
    }
};

export const fetchFeatures = async () => {
    try {
        const response = await fetch(window.API_URL + "/features_list", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });

        return response.ok ? await response.json() : [];
    } catch (error) {
        console.error("Error fetching features:", error);
        return [];
    }
};

export const fetchClassificationCount = async (token) => {
    try {
        const response = await fetch(window.API_URL + "/count_tags_made_by_user", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token,
            },
        });

        return response.ok ? await response.json() : null;
    } catch (error) {
        console.error("Error fetching classification count:", error);
        return null;
    }
};

export const fetchUserStats = async (token) => {
    try {
        const response = await fetch(window.API_URL + "/get_user_panel", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token,
            },
        });
        return response.ok ? await response.json() : null;

    } catch (error) {
        console.error("Error fetching user stats:", error);
        return null;
    }
};

