$(document).ready(function () {
    var is_ok = "N";

    requestData();

    function requestData() {
        if (is_ok === "N") {
            loadDoc();
            setTimeout(requestData, 1000);
        }
    }

    function loadDoc() {
        $.ajax({
            type: "POST",
            url: "/get_data",
        }).done(function (data) {
            if (parseInt(data.sts) === 2) {
                is_ok = "Y";
                window.location.href = "/facerecog/" + data;
            }
        });
    }

    $("#search-form").on("submit", function (event) {
        event.preventDefault();

        var formData = $(this).serialize();

        $.ajax({
            data: formData,
            type: "POST",
            url: "/search",
        }).done(function (data) {
            displaySearchResults(data);
        });
    });

    function displaySearchResults(data) {
        var resultsDiv = $("#search-results");
        resultsDiv.empty();

        if (data.error) {
            resultsDiv.append("<p>Error: " + data.error + "</p>");
            return;
        }

        var table = $("<table>");
        table.append("<thead><tr><th>Prs Number</th><th>Prs Name</th><th>Accs Date</th></tr></thead>");

        var tbody = $("<tbody>");

        $.each(data, function (index, row) {
            var tr = $("<tr>");
            tr.append("<td>" + row.student_id + "</td>");
            tr.append("<td>" + row.student_name + "</td>");
            tr.append("<td>" + row.enter_date + ", " + row.out_date + "</td>");
            tbody.append(tr);
        });

        table.append(tbody);
        resultsDiv.append(table);
    }
});