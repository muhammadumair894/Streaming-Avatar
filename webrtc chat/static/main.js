let pc = new RTCPeerConnection();
let dc = pc.createDataChannel("chat");

dc.onopen = () => console.log("Data channel is open");
dc.onmessage = event => {
    console.log(`Received message: ${event.data}`);
    let li = document.createElement("li");
    li.textContent = event.data;
    document.getElementById("messages").appendChild(li);
};

pc.ondatachannel = event => {
    const receiveChannel = event.channel;
    receiveChannel.onmessage = event => {
        console.log(`Received message: ${event.data}`);
    };
    receiveChannel.onopen = () => console.log("Received data channel is open");
};

document.getElementById("send").addEventListener("click", async function(event) {
    event.preventDefault(); 

    const message = document.getElementById("input").value;
    if (dc.readyState === "open") {
        dc.send(message);
        console.log(`Message sent: ${message}`);

        document.getElementById("input").value = ""; 

        const chatBox = document.getElementById("messages");
        let li = document.createElement("li");
        li.textContent = `You: ${message}`;
        chatBox.appendChild(li);
    } else {
        console.log("Data channel is not open.");
    }
});

async function createOffer() {
    console.log("Creating offer...");
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    console.log("Offer created and set as local description");

    const offerResponse = await fetch("/offer", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type,
        }),
        
        
    });
    
    const answer = await offerResponse.json();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));
    console.log("Answer set as remote description");
}


createOffer();
