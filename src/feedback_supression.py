#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK


'''Feedback suppression using frequency domain processing.'''

import numpy as np
import logging
import scipy.fft as fft
import sounddevice as sd

import minimal
import buffer
        
class Feedback_Supression(buffer.Buffering):
    def __init__(self):
        super().__init__()
        logging.info(__doc__)
        
        # Initialize FFT parameters
        self.fft_size = minimal.args.frames_per_chunk
        self.freq_bins = np.arange(self.fft_size // 2)
        
        # Feedback detection and suppression parameters
        self.feedback_threshold = 1  # Threshold for detecting feedback
        self.suppression_factor = 1  # Amount to suppress feedback
        
    def apply_feedback_suppression(self, chunk):
        """Apply frequency domain feedback suppression."""
        # Convert to frequency domain using FFT
        fft_chunk = fft.fft(chunk, axis=0)
        
        # Create a mask for frequency bins that might contain feedback
        # This is a simple implementation - you can enhance based on your specific needs
        feedback_mask = np.abs(fft_chunk) > self.feedback_threshold * np.max(np.abs(fft_chunk))
        
        # Apply suppression to detected feedback components
        suppressed_fft = fft_chunk.copy()
        suppressed_fft[feedback_mask] *= (1 - self.suppression_factor)
        
        # Convert back to time domain
        suppressed_chunk = fft.ifft(suppressed_fft, axis=0).real
        
        return suppressed_chunk.astype(np.int16)

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        """Override the recording callback with feedback suppression."""
        # Apply feedback suppression to the incoming chunk
        suppressed_ADC = self.apply_feedback_suppression(ADC)
        
        # Pack and send the suppressed chunk
        packed_chunk = self.pack(self.chunk_number, suppressed_ADC)
        self.send(packed_chunk)
        
        # Receive and play the chunk (same as parent class)
        chunk = self.unbuffer_next_chunk()
        self.play_chunk(DAC, chunk)

    def _read_IO_and_play(self, DAC, frames, time, status):
        """Override the file reading callback with feedback suppression."""
        # Read a chunk from file
        read_chunk = self.read_chunk_from_file()
        
        # Apply feedback suppression to the read chunk
        suppressed_chunk = self.apply_feedback_suppression(read_chunk)
        
        # Pack and send the suppressed chunk
        packed_chunk = self.pack(self.chunk_number, suppressed_chunk)
        self.send(packed_chunk)
        
        # Receive and play the chunk (same as parent class)
        chunk = self.unbuffer_next_chunk()
        self.play_chunk(DAC, chunk)
        return read_chunk

class Feedback_Supression__verbose(Feedback_Supression, buffer.Buffering__verbose):
    def __init__(self):
        super().__init__()

try:
    import argcomplete  # <tab> completion for argparse.
except ImportError:
    logging.warning("Unable to import argcomplete (optional)")

if __name__ == "__main__":
    minimal.parser.description = __doc__
    try:
        argcomplete.autocomplete(minimal.parser)
    except Exception:
        logging.warning("argcomplete not working :-/")
    minimal.args = minimal.parser.parse_known_args()[0]

    if minimal.args.show_stats or minimal.args.show_samples or minimal.args.show_spectrum:
        intercom = Feedback_Supression__verbose()
    else:
        intercom = Feedback_Supression()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
       intercom.print_final_averages()
