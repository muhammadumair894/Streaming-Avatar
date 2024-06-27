from flask import Flask, render_template, request, jsonify
from aiortc import RTCPeerConnection, RTCSessionDescription

app = Flask(__name__)
data_channels = {}  # Dictionary to store data channels per connection
pcs = set()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/offer', methods=['POST'])
async def offer():
    params = request.json
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
    pc = RTCPeerConnection()
    pcs.add(pc)

    dc = pc.createDataChannel('chat')
    data_channels[pc] = dc

    @dc.on('message')     
    async def on_message(message):
      # Get the data channel for the other peer
      other_pc = next(pc for pc in data_channels if pc != dc.pc) 
      other_channel = data_channels[other_pc]
  
      if other_channel.readyState == "open":
        await other_channel.send(message)
      else:
        print(f"Data channel for peer {other_pc} is not open.")


    try:
      await pc.setRemoteDescription(offer)
      answer = await pc.createAnswer()
      await pc.setLocalDescription(answer)
      return jsonify({'sdp': answer.sdp, 'type': answer.type})
    except Exception as e:
      print(f"Error: {e}")
      return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
