sleep 5
while sleep 1; do
    python3 /home/pi/Desktop/ejecutables/timestamped.py
    now=$(date +"%T")
    echo "Crash time : $now"
    echo "reconocimiento.py crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
