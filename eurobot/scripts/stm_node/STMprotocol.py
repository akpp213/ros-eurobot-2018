import serial
import struct
import datetime


class STMprotocol(object):
    def __init__(self, serial_port):
        self.ser = serial.Serial(serial_port, baudrate=115200, timeout=0.2)
        self.pack_format = {
            0x01: "=BBBB",
            0x03: "=Bf",
            0x04: "=B",
            0x05: "=B",
            0x08: "=fff",
            0x09: "=",
            0x0a: "=",
            0x0b: "=BH",
            0x0c: "=B",
            0x0d: "=B",
            0xa0: "=fff",
            0xa1: "=fff",
            0xb0: "=B",
            0xc0: "=BB",
            0xb1: "=B",
            0x0e: "=fff",
            0x0f: "=",
        }

        self.unpack_format = {
            0x01: "=BBBB",
            0x03: "=BB",
            0x04: "=BB",
            0x05: "=BB",
            0x08: "=BB",
            0x09: "=fff",
            0x0a: "=fff",
            0x0b: "=BB",
            0x0c: "=f",
            0x0d: "=BB",
            0xa0: "=Bfff",
            0xa1: "=BB",
            0xb0: "=BB",
            0xc0: "=BB",
            0xb1: "=BB",
            0x0e: "=BB",
            0x0f: "=fff",
        }
    def confident_send_command(self, cmd, args, n_repeats):
        for i in range(n_repeats):
            pass
    def send_command(self, cmd, args):
        try:
            parameters = bytearray(struct.pack(self.pack_format[cmd], *args))
            #print(parameters)
            msg_len = len(parameters) + 5
            msg = bytearray([0xfa, 0xaf, msg_len, cmd]) + parameters
            crc = sum(msg) % 256
            msg += bytearray([crc])

            #print("send ", repr(msg))
            self.ser.write(msg)

            start_time = datetime.datetime.now()
            time_threshold = datetime.timedelta(seconds=1)
            dt = start_time - start_time

            data = ord(self.ser.read()[0])
            while data != 0xFA:
                if dt > time_threshold:
                    raise Exception('dt > threshold')
                data = ord(self.ser.read()[0])

                current_time = datetime.datetime.now()
                dt = start_time - current_time

            adr = ord(self.ser.read()[0])
            answer_len = ord(self.ser.read()[0])
            answer = bytearray(self.ser.read(answer_len - 3))
            #print("answer ", repr(bytearray([data, adr, answer_len]) + answer))

            args = struct.unpack(self.unpack_format[cmd], answer[1:-1])
            #print 'SUCCESS'
            #print '-------------------------------'
            return True, args
        except Exception as exc:
            print 'Exception:\t', exc
            print 'Of type:\t', type(exc)
            print 'At time:\t', datetime.datetime.now()
            print("send ", repr(msg))
            print("answer ", repr(bytearray([data, adr, answer_len]) + answer))
            print '--------'
            return False, 0
