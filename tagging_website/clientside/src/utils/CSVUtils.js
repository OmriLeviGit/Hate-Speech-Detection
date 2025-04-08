export const exportToCSV = (users) => {
    if (!users || users.length === 0) {
        alert("No data to export!");
        return;
    }

    const headers = [
        "Email,No. Classified, Left to Classify, Average Time (seconds), No. Positive,No. Negative,No. Irrelevant,No. Uncertain,% Positive,% Negative,% Irrelevant,% Uncertain"
    ];
    const rows = users.map(user => [
        user.email,
        user.personalClassifications,
        user.leftToClassify,
        user.averageTime > 0 ? user.averageTime : 0,
        user.positiveClassified,
        user.negativeClassified,
        user.irrelevantClassified,
        'number uncertain',
        user.personalClassifications > 0 ? ((user.positiveClassified / user.personalClassifications) * 100).toFixed(2) : 0,
        user.personalClassifications > 0 ? ((user.negativeClassified / user.personalClassifications) * 100).toFixed(2) : 0,
        user.personalClassifications > 0 ? ((user.irrelevantClassified / user.personalClassifications) * 100).toFixed(2) : 0,
        '% Uncertain'
    ].join(","));

    const csvContent = [headers, ...rows].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;

    // Creating the timestamp to attach to the file's name
    const now = new Date();
    const formattedTimestamp = `${now.getDate()}_${now.getMonth() + 1}_${now.getFullYear()}_${now.getHours()}_${now.getMinutes()}`;
    link.download = `user_stats_${formattedTimestamp}.csv`;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};