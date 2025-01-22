from flask import Flask, request, jsonify, send_file
import test_lambert
import numpy as np

app = Flask(__name__)

@app.route('/trajectory', methods=['POST'])
def trajectory():
    try:
        data = request.get_json()
        r1 = np.array(data['r1'])
        r2 = np.array(data['r2'])
        tof = data['tof']
        start_time = data['start_time']

        test_lambert.main(r1, r2, tof, start_time)
        send_file('trajectory.csv', as_attachment=True), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

