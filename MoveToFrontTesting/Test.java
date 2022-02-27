
import java.io.File;
import java.io.FileInputStream;
import java.lang.Exception;
import java.util.HashMap;
import java.util.Map;

class Test {
    static int[] original;
    static int[] encoded; 
    static int[] out;
    static long encodingDuration;
    static long decodingDuration;
    static String dataStructureType;
	public static void main(String[] args) throws Exception { 
        // Init data structures to be used.
        Map<String, Runnable> dataStructureTypes = new HashMap<>();

        dataStructureTypes.put("ll", () -> {
            encodingDuration = MoveToFrontLinkedList.testEncode(original, encoded);
            decodingDuration = MoveToFrontLinkedList.testDecode(encoded, out);
        });
        dataStructureTypes.put("al", () -> {
            encodingDuration = MoveToFrontArrayList.testEncode(original, encoded);
            decodingDuration = MoveToFrontArrayList.testDecode(encoded, out);
        });
        dataStructureTypes.put("alR", () -> {
            encodingDuration = MoveToFrontArrayListReversed.testEncode(original, encoded);
            decodingDuration = MoveToFrontArrayListReversed.testDecode(encoded, out);
        });
        dataStructureTypes.put("ar", () -> {
            encodingDuration = MoveToFrontArray.testEncode(original, encoded);
            decodingDuration = MoveToFrontArray.testDecode(encoded, out);
        });
        dataStructureTypes.put("arR", () -> {
            encodingDuration = MoveToFrontArrayReversed.testEncode(original, encoded);
            decodingDuration = MoveToFrontArrayReversed.testDecode(encoded, out);
        });

        // Execution of the file from command prompt.
		if (args.length == 0) {
            throw new Exception("Not enough arguments");
        }

        if (args[0].equals("help")) {
            System.out.println("test [dataStructureType] -f [filename]");
            System.out.println("test [dataStructureType] -n [random chars amount]");
            System.out.println("dataStructureTypes: ll, al, alR, ar, arR");
            return;
        } else if (args[0].equals("test")) {
            if (!dataStructureTypes.containsKey(args[1])) {
                throw new Exception("No such data structure type");
            }
            dataStructureType = args[1];
        } else {
            throw new Exception("Wrong call");
        }

        if (args[2].equals("-f")) {
            try (FileInputStream inFile = new FileInputStream(args[3])) {
                int byteRead;
                int i = 0;
                int bytesAmount = (int) inFile.getChannel().size();

                // Init array with fixed length
                original = new int[bytesAmount];
                encoded = new int[bytesAmount];
                out = new int[bytesAmount];

                while ((byteRead = inFile.read()) != -1) {
                    original[i] = byteRead;
                    encoded[i] = 0;
                    out[i] = 0;
                    i++;
                }
            }

        } else if (args[2].equals("-n")) {
            int bytesAmount = Integer.parseInt(args[3]);

            // Init array with fixed length
            original = new int[bytesAmount];
            encoded = new int[bytesAmount];
            out = new int[bytesAmount];

            for (int i = 0; i < bytesAmount; i++) {
                original[i] = (int) Math.round(Math.random() * 255); // Integer between 0 and 255
                encoded[i] = 0;
                out[i] = 0;
            }
        }
        
        // Run the specific encoding / decoding test. 
        // This changes the variables encodingDuration and decodingDuration.
        dataStructureTypes.get(dataStructureType).run();

        // Check for correctness after encoding and decoding.
        for (var i = 0; i < original.length; i++) {
            if (original[i] != out[i]) {
                throw new Exception("Not same array after encoding and decoding");
            }
        }
        //System.out.println("Using " + dataStructureType +  ": " + " encoding: " + encodingDuration/1e+9 + " seconds, decoding: " + decodingDuration/1e+9 + " seconds");
        System.out.println(encodingDuration + " " + decodingDuration);
	}
}
