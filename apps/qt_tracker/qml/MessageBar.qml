import QtQuick
import QtQuick.Controls

Label {
    id: bar

    function show(message) {
        text = message
        visible = true
        hideTimer.restart()
    }

    visible: false
    padding: 10
    color: "white"
    background: Rectangle { color: "#dd333333"; radius: 6 }

    Timer {
        id: hideTimer
        interval: 4000
        onTriggered: bar.visible = false
    }
}
