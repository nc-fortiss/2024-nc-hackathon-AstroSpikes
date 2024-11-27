#!/bin/bash

# Configuration
SCRIPTNAME="cron_start_training.sh"          # default is cron_start_training.sh
CRONDIR="/etc/cron.hourly"                   # Standard directory for hourly cron jobs

# Check if running with sudo/root permissions
if [ "$EUID" -ne 0 ]; then
    echo "Please run this script with sudo permissions"
    exit 1
fi

# Ensure the cron directory exists
if [ ! -d "$CRONDIR" ]; then
    echo "Creating cron directory: $CRONDIR"
    mkdir -p "$CRONDIR"
fi

# Copy the script to cron directory
echo "Copying $SCRIPTNAME to $CRONDIR"
cp "$SCRIPTNAME" "$CRONDIR"

# Make the script executable
echo "Making script executable"
chmod +x "$CRONDIR/$SCRIPTNAME"

# Verify installation
if [ -x "$CRONDIR/$SCRIPTNAME" ]; then
    echo "Successfully installed $SCRIPTNAME in $CRONDIR"
    echo "Script will run hourly"
else
    echo "Failed to install script"
    exit 1
fi

# Restart cron service to ensure changes take effect
echo "Restarting cron service"
if command -v systemctl &> /dev/null; then
    systemctl restart cron
else
    service cron restart
fi

echo "Setup completed successfully"