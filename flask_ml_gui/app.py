from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# ML models
models = {
    "Linear Regression": joblib.load("LinRegModel.joblib"),
    "Ridge Regression": joblib.load("RidgeRegModel.joblib"),
    "Random Forest": joblib.load("RandForestModel.joblib")
}

scaler = joblib.load('scaler.joblib')

@app.route("/", methods=["GET", "POST"])
def index():

    results = None
    team_a = ""
    team_b = ""

    if request.method == "POST":

        # team names
        team_a = request.form["team_a_name"]
        team_b = request.form["team_b_name"]

        # team A stats
        a_orb = float(request.form["a_orb"])
        a_drb = float(request.form["a_drb"])
        a_ast = float(request.form["a_ast"])
        a_blk = float(request.form["a_blk"])
        a_tov = float(request.form["a_tov"])
        a_pf = float(request.form["a_pf"])


        skip_b = False
        if not team_b:
            skip_b = True

        # team B stats
        if not skip_b:
            b_orb = float(request.form["b_orb"])
            b_drb = float(request.form["b_drb"])
            b_ast = float(request.form["b_ast"])
            b_blk = float(request.form["b_blk"])
            b_tov = float(request.form["b_tov"])
            b_pf = float(request.form["b_pf"])

        #Flag for if there's only 1 team inputted
        

        # compute diff in features (Team A - Team B)
        a_stats = [a_orb, a_drb, a_ast, a_blk, a_tov, a_pf]
        if not skip_b:
            b_stats = [b_orb, b_drb, b_ast, b_blk, b_tov, b_pf]


        #Change into floats
        a_stats = [float(i) for i in a_stats]
        a_stats = np.array(a_stats)
        X_a = a_stats.reshape(1, -1)
        X_a = scaler.transform(X_a)
        

        if not skip_b:
            b_stats = [float(i) for i in b_stats]
            b_stats = np.array(b_stats)
            X_b = b_stats.reshape(1, -1)
            X_b = scaler.transform(X_b)

        # Convert to sklearn input shape
        
        

        results = []

        # get prediction from each model
        for name, model in models.items():
            point_diff_a = model.predict(X_a)
            if not skip_b:
                point_diff_b = model.predict(X_b)
            else:
                point_diff_b = 'N/A'
            if skip_b:
                winner = 'N/A'
            else:
                if point_diff_a - point_diff_b > 0:
                    winner = team_a
                else:
                    winner = team_b
            if not skip_b:
                results.append({
                    "model": name,
                    "apointdiff": np.round(point_diff_a, decimals = 2),
                    "bpointdiff": np.round(point_diff_b, decimals = 2),
                    "netdiff": np.round(point_diff_a - point_diff_b, decimals = 2),
                    "winner": winner
                })
            else:
                results.append({
                    "model": name,
                    "apointdiff": np.round(point_diff_a, decimals = 2),
                    "bpointdiff": 'N/A',
                    "netdiff": 'N/A',
                    "winner": winner
                })
    
    # render updates into index.html
    return render_template(
        "index.html",
        results=results,
        team_a=team_a,
        team_b=team_b
    )


if __name__ == "__main__":
    app.run(debug=True)