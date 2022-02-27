import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.fingerprint.*;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.smiles.*;
import org.openscience.cdk.silent.*;
import org.openscience.cdk.similarity.*;
import java.io.*;
import java.util.BitSet;

public class TanimotoCalculate {
public static void main(String []args) throws CDKException, IOException {
    String dirPath="C:\\Users\\Administrator\\Desktop\\新建文件夹\\";
    String testHypethese ="C:\\Users\\XuZhanPeng\\Desktop\\pre.txt";
    String testTruelabel ="C:\\Users\\XuZhanPeng\\Desktop\\ture.txt";
    String fileWrite="C:\\Users\\XuZhanPeng\\Desktop\\result_swin.txt";
    Double similarity = 0.0;
    int rightNum = 0,errNum=0;
    File record = new File(fileWrite);//记录结果文件
    FileWriter writer = new FileWriter(record, true);
    BufferedReader enBr = new BufferedReader(new FileReader(testHypethese));
    BufferedReader frBr = new BufferedReader(new FileReader(testTruelabel));
    SmilesParser smilesParser = new SmilesParser(
            SilentChemObjectBuilder.getInstance()
    );
    while(true){
        String smilesHypethese = enBr.readLine();
        String smilesTruelabel = frBr.readLine();
        if(smilesHypethese == null || smilesTruelabel == null){
            break;
        }
//        System.out.println(smilesHypethese +"\t"+ smilesTruelabel);
        String smiles1 = smilesHypethese;
        String smiles2 = smilesTruelabel;
        try{

            IAtomContainer mol1 = smilesParser.parseSmiles(smiles1);
            IAtomContainer mol2 = smilesParser.parseSmiles(smiles2);
            HybridizationFingerprinter fingerprinter = new HybridizationFingerprinter();
            BitSet bitset1 = fingerprinter.getFingerprint(mol1);
            BitSet bitset2 = fingerprinter.getFingerprint(mol2);
            double tanimoto = Tanimoto.calculate(bitset1, bitset2);
            if(Double.isNaN(tanimoto)){
                tanimoto=0.0;
            }
            similarity+=tanimoto;
            rightNum++;

//            System.out.println ("Tanimoto: "+tanimoto);
//            System.out.println ("similarity: "+similarity);

        }catch(Exception e){
            errNum++;
//            System.out.println ("Error！");
        }
    }
    System.out.print("Sililarity:"+(similarity/rightNum));
    writer.write("\n"+testHypethese+"\n"+"Sililarity:"+(similarity/rightNum)+"\nRight:"+rightNum+"\nError:"+errNum+
            "\nRightParse："+((float)rightNum/(rightNum+errNum)));
    writer.close();
}
}
